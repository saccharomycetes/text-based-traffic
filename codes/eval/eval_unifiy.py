from torch.cuda import reset_max_memory_allocated
import transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from sentence_transformers import SentenceTransformer, util
import os
import numpy as np
import json
from tqdm import tqdm
import argparse
import difflib
import random

def string_similar(s1, s2):
    return difflib.SequenceMatcher(None, s1, s2).quick_ratio()


torch.cuda.empty_cache()
def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return int(np.argmax(memory_available))

class dpr_uni_model():

    def __init__(self, args):
        self.args = args
        with open(self.args.corpus_file)as f:
            paras = json.load(f)
        self.passages = [i for j in paras.keys() for k in paras[j].keys() for i in paras[j][k]]
        self.tokenizer = T5Tokenizer.from_pretrained(self.args.model_type)
        self.model = T5ForConditionalGeneration.from_pretrained(self.args.model_type)
        self.passage_encoder = SentenceTransformer('facebook-dpr-ctx_encoder-single-nq-base')
        self.query_encoder = SentenceTransformer('facebook-dpr-question_encoder-single-nq-base')
        self.model.to(self.args.device)
    
    def construct_input(self, question, candidates, sort_index, passages):
        input_string = question + "\\n"
        options = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"]
        for i, ans in enumerate(candidates):
            input_string += " ("+options[i]+") "+ ans
        if self.args.num_related > 0:
            input_string += "\\n "
            for i in sort_index[:self.args.num_related]:
                input_string += passages[i] + " "
        input_id = self.tokenizer.encode(input_string, return_tensors="pt").to(self.args.device)
        return input_id

    def load_data(self):
        with open(self.args.dev_file)as f:
            datas = f.readlines()
        raw_datas = [json.loads(data) for data in datas]
        if self.args.data == "bdd":
            if self.args.test_type == "e":
                question = "c"
                candidate = "e"
            else:
                question = "e"
                candidate = "c"
        else:
            question = "question"
            candidate = "candidates"
        datas = [{"question":  data[question],
                  "candidate": data[candidate],
                  "answer":    data["answer"],
                  "id":        data["id"]} for data in raw_datas[:5000]]
        # uniform the data

        with open(self.args.corpus_file)as f:
            paras = json.load(f)
        passages = [i for j in paras.keys() for k in paras[j].keys() for i in paras[j][k]]
        return datas, passages
    
    def evaluate(self):
        datas, passages = self.load_data()
        if self.args.num_related > 0:
            passage_embeddings = self.passage_encoder.encode(passages, batch_size=32, show_progress_bar=True)

        pred_answers = []
        if not os.path.exists(self.args.output_dir):
            os.makedirs(self.args.output_dir)
        
        for data in tqdm(datas, ncols=100):
            if self.args.num_related > 0:
                q_embedding = self.query_encoder.encode(data["question"])
                related_scores = util.dot_score(q_embedding, passage_embeddings)
                sort_index = torch.sort(related_scores, dim=1, descending=True)[1][0].tolist()
            else:
                sort_index = []

            input_id = self.construct_input(data["question"], data["candidate"], sort_index, passages)
            pred_id = self.model.generate(input_id)
            pred_answer = self.tokenizer.batch_decode(pred_id, skip_special_tokens=True)[0]

            pred_answers.append(pred_answer)

        similaritiess = np.array([[string_similar(i, k) for k in j["candidate"]] for i, j in zip(pred_answers, datas)])
        
        preds = [int(i) for i in (np.argmax(similaritiess, axis=1))]
        corrects = [data["answer"] for data in datas]
        ids = [data["id"] for data in datas]

        rights = [k for i, j, k in zip(preds, corrects, ids) if i==j]
        acc = len(rights)/len(ids)

        output_acc_file = os.path.join(self.args.output_dir, "acc.txt")
        with open(output_acc_file, "w") as writer:
            print("***** Eval results *****")
            print("pas_num:" + str(self.args.num_related))
            print("  acc = %s", str(acc))
            writer.write("acc = %s\n" % (str(acc)))
        preds_info = {"preds": preds, "rights": rights}
        output_preds_file = os.path.join(self.args.output_dir, "preds.json")
        with open(output_preds_file, "w") as f:
            json.dump(preds_info ,f)
        return acc

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="hdt", type=str, help="The data name")
    parser.add_argument("--test_type", default="e", type=str, help="type of testing data")
    parser.add_argument("--num_related", default=1, type=int, help="The eval file name")
    args = parser.parse_args()

    args.output_dir = "../../results/dpr/pas" + str(args.num_related) + "/" + args.data + "_" + args.test_type

    args.dev_file = "../../data/" + args.data + "/" + args.test_type + ".jsonl"

    args.corpus_file = "../../data/pas/paras.json"

    args.model_type = "allenai/unifiedqa-v2-t5-3b-1251000"
    # args.model_type = "allenai/unifiedqa-v2-t5-small-1363200"

    free_gpu = get_freer_gpu()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(free_gpu)
    args.device = device

    model = dpr_uni_model(args)

    result = model.evaluate()
    return result


if __name__ == "__main__":
	main()