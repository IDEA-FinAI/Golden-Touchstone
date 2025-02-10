import json
import os
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from seqeval.metrics import f1_score as entity_f1
import random
from nltk.translate.bleu_score import sentence_bleu,SmoothingFunction
from rouge_chinese import Rouge
import jieba
import numpy as np
from tqdm import tqdm


#英文RE评测
class RE_en_eval:
    def __init__(self) -> None:
        
        self.relations = [
            'product_or_material_produced',
            'manufacturer',
            'distributed_by',
            'industry',
            'position_held',
            'original_broadcaster',
            'owned_by',
            'founded_by',
            'distribution_format',
            'headquarters_location',
            'stock_exchange',
            'currency',
            'parent_organization',
            'chief_executive_officer',
            'director_/_manager',
            'owner_of',
            'operator',
            'member_of',
            'employer',
            'chairperson',
            'platform',
            'subsidiary',
            'legal_form',
            'publisher',
            'developer',
            'brand',
            'business_division',
            'location_of_formation',
            'creator',
        ]
    def cpt_label_and_pred(self,text_pred,text_label,prompt):
        tp,fp,fn = 0,0,0
        
        choice = re.search('relation1: word1, word2',prompt)
        text_pred_sub = text_pred.strip('.').split(';')
        text_label_sub = text_label.strip('.').split(';')
        for pred_txt,label_txt in zip(text_pred_sub,text_label_sub):
            if choice:
                pred_match = re.search(r'^(.*?):(.*?)?,(.*)?$', pred_txt)
                try:
                    pred_match = [pred_match.group(1),pred_match.group(2),pred_match.group(3)]
                except BaseException:
                    pred_match = ['','','']
                label_match = re.search(r'^(.*?):(.*?)?,(.*)?$', label_txt)
                label_match = [label_match.group(1),label_match.group(2),label_match.group(3)]
            else:
                pred_match = re.search(' '.join(label_txt.split('_')),pred_txt.split(',')[0])
                if not pred_match:
                    pred_match = ['unknown']
                else:
                    pred_match = [pred_match.group(0)]
                label_match = [label_txt]
                
            pred_match = list(pred_match)
            label_match = list(label_match)

            pred_match[0] = ' '.join(pred_match[0].split('_'))
            label_match[0] = ' '.join(label_match[0].split('_'))

            for i in range(len(pred_match)):
                pred_match[i] = pred_match[i].strip()
            
            for i in range(len(label_match)):
                label_match[i] = label_match[i].strip()
                


            # 计算 TP（预测正确的关系）pred 与 label 相同
            flagTP = True
            for i in range(len(pred_match)):
                if pred_match[i] != label_match[i]:
                    flagTP = False
            if flagTP:
                tp += 1
            else:
                fp += 1


            # 计算 FN（漏掉的正确关系）遗漏了一个真实存在的关系
            if not('_'.join(pred_match[0].split()) in self.relations):
                fn += 1
 
        return tp, fp, fn

    def get_result(self,file_path,output_path):
        tp_all,fp_all,fn_all = 0,0,0
        task = file_path.split('/')[-1]
        file_path = os.path.join(file_path,'generated_predictions.jsonl')
        with open(file_path,'r',encoding='utf-8') as f:
            contents = json.load(f)
            for content in contents:
                # if content['label'] == preprocess_output(content['predict']):
                tp,fp,fn = self.cpt_label_and_pred(content['predict'],content['label'],content['prompt'])
                tp_all += tp
                fp_all += fp
                fn_all += fn
                
        # 计算 precision, recall, F1 score
        precision = tp_all / (tp_all + fp_all) if (tp_all + fp_all) > 0 else 0
        recall = tp_all / (tp_all + fn_all) if (tp_all + fn_all) > 0 else 0
        Relation_f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        result = {
                    'precision':precision,
                    'recall':recall,
                    'Relation_f1_score':Relation_f1_score
                }
        with open(os.path.join(output_path,f'{task}_result.json'),'w',encoding='utf-8') as f:
            json.dump(result,f,ensure_ascii=False)
        return Relation_f1_score

#英文QA评测
class QA_en_eval:

    def preprocess_output(self,pred,label):
        label = label.strip().lower().split('the answer is:')[-1]
        pred = pred.strip().lower()
        temp = re.search(label,pred)

        pred = label if temp else "unknown"
        return pred, label


    def get_result(self,file_path,output_path):
        preds = []
        labels = []
        task = file_path.split('/')[-1]
        file_path = os.path.join(file_path,'generated_predictions.jsonl')

        with open(file_path,'r',encoding='utf-8') as f:
            contents = json.load(f)
            for content in contents:
                pred, label = self.preprocess_output(content['predict'],content['label'])

                preds.append(pred)
                labels.append(label)

        accuracy = accuracy_score(labels, preds)

        with open(os.path.join(output_path,f'{task}_result.json'),'w',encoding='utf-8') as f:
            json.dump({"accuracy":accuracy},f,ensure_ascii=False)
        return accuracy


#英文Ner评测
class Ner_en_eval:
    def __init__(self) -> None:
          self.arrtibute_list = ['ORG','LOC','PER']

    #tokens是predict的实体
    def cvt_text_to_pred(self, tokens, text):

        tokens = tokens.strip().lower().split()
        word_preds = ['O' for _ in range(len(tokens))]
        pattern = r"([\w\s]+),\s*([A-Z]+)"
        matches = re.findall(pattern, text)
        for name, arrtribute in matches:
            entity_tokens = name.strip().lower().split()
            if not arrtribute in self.arrtibute_list:
                arrtribute = self.arrtibute_list[0]
            n = len(entity_tokens)
            if n == 0:
                continue
            for i in range(len(tokens) - n + 1):
                    if tokens[i:i+n] == entity_tokens and word_preds[i:i+n] == ['O'] * n:
                        word_preds[i:i+n] = ['B-' + arrtribute] + ['I-' + arrtribute] * (n-1)
                        break

        return word_preds


    def get_result(self, file_path, output_path):
        preds = []
        labels = []
        texts = []
        task = file_path.split('/')[-1]
        file_path = os.path.join(file_path,'generated_predictions.jsonl')
        #打开原始ner文件，收集label和text信息
        with open('/finance_ML/FinLLM/align_version_FinLLM-benchmark/EN-benchmark/flare-ner/flare-ner_test.json','r') as f:
            lines = json.load(f)
            for line in lines:
                texts.append(line['input'])
                labels.append(line['label'])


        with open(file_path,'r',encoding='utf-8') as f:
            contents = json.load(f)
            for id,content in enumerate(contents):
                preds.append(self.cvt_text_to_pred(texts[id],content['predict']))

        assert len(labels) == len(preds)
        # for id,item in enumerate(labels):
        #     if item != preds[id]:
        #         print(id)
        f1 = entity_f1(labels, preds, average='weighted')
        with open(os.path.join(output_path,f'{task}_result.json'),'w',encoding='utf-8') as f:
                json.dump({"weighted-F1":f1},f,ensure_ascii=False)
        return f1

#英文分类评测
class classify_en_eval:
    def __init__(self) -> None:
        self.tasks_enable = {'flare-fpb':self.preprocess_output_sentiment,
                             'fiqasa':self.preprocess_output_sentiment,
                             'flare-headlines':self.preprocess_output_headline,
                             'flare-fomc':self.preprocess_output_fomc,
                             'CFA-multiple_choice':self.preprocess_output_choice,
                             'DJIA_stock_prediction': self.preprocess_output_stock,
                             'cra-lendingclub':self.preprocess_output_credit,
                             'CMIN-US-Windows-test':self.preprocess_output_stock
                             }
        self.stock_dict = {'rise':'1','decrease':'0'}
        self.task_label_enable = {
                                'flare-fpb':["positive","negative","neutral"],
                                'fiqasa':["positive","negative","neutral"],
                                'flare-headlines':["yes","no"],
                                'flare-fomc':["hawkish","dovish","neutral"],
                                'CFA-multiple_choice':["a","b","c","d"],
                                'DJIA_stock_prediction':["0","1"],
                                'cra-lendingclub':["good","bad"],
                                'CMIN-US-Windows-test':["0",'1']
                             }
    def preprocess_output_sentiment(self,output):
        output = output.strip().lower()
        answer_match = re.search(r'\b(positive|negative|neutral)\b',output)
        if answer_match:
            temp = answer_match.group(1)
        else:
            temp = "unknown"
        return temp

    def preprocess_output_headline(self,output):
        output = output.strip().lower()
        answer_match = re.search(r'\b(yes|no)\b',output)
        if answer_match:
            temp = answer_match.group(1)
        else:
            temp = "unknown"
        return temp

    def preprocess_output_credit(self,output):
        output = output.strip().lower()

        answer_match = re.search(r'\b(good|bad)\b',output)

        if answer_match:
            temp = answer_match.group(1)
        else:
            temp = "unknown"
        return temp

    def preprocess_output_fomc(self,output):
        output = output.strip().lower()

        answer_match = re.search(r'\b(hawkish|dovish|neutral)\b',output)

        if answer_match:
            temp = answer_match.group(1)
        else:
            temp = "unknown"
        return temp

    def preprocess_output_choice(self,output):
        output = output.strip().lower()
        answer_match = re.search(r'\b(a|b|c|d)\b',output)

        if answer_match:
            temp = answer_match.group(1)
        else:
            temp = "unknown"
        return temp

    def preprocess_output_stock(self,output):
        output = output.strip().lower()

        answer_match = re.search(r'\b(0|1)\b',output)
        
        if answer_match:
            temp = answer_match.group(1)
        else:
            answer_match = re.search(r'\b(rise|decrease)\b',output)
            if answer_match:
                temp = self.stock_dict[answer_match.group(1)]
            else:
                temp = "unknown"
        return temp


    def get_result(self,file_path,output_path):

        pred = []
        label = []
        task = file_path.split('/')[-1]
        file_path = os.path.join(file_path,'generated_predictions.jsonl')
        preprocess_output = self.tasks_enable[task]
        with open(file_path,'r',encoding='utf-8') as f:
            contents = json.load(f)
            for content in contents:
                pred_temp = preprocess_output(content['predict'])
                label_temp = preprocess_output(content['label'])

                if pred_temp == "unknown":
                    label_dict = self.task_label_enable[task][:]
                    for i in range(len(label_dict)):
                        if label_temp == label_dict[i]:
                            label_dict.pop(i)
                            break
                    pred_temp = random.choice(label_dict)

                pred.append(pred_temp)
                label.append(label_temp)
        labels = list(set(label))
        accuracy = accuracy_score(label, pred)
        precision = precision_score(label, pred, average="weighted")
        recall = recall_score(label, pred, average="weighted",zero_division=0)
        f1 = f1_score(label, pred, average="weighted",labels=labels)
        mcc = matthews_corrcoef(label, pred)
        result = {
                'accuracy':accuracy,
                'precision':precision,
                'recall':recall,
                'weighted-F1':f1,
                'MCC':mcc
            }
        with open(os.path.join(output_path,f'{task}_result.json'),'w',encoding='utf-8') as f:
            json.dump(result,f,ensure_ascii=False)
        
        if task == 'cra-lendingclub' or task == 'DJIA_stock_prediction':
            return (f1+mcc)/2
        else:
            return (accuracy+f1) / 2
# QA评测
class QA_cn_eval:
    def preprocess_output_qa_cn(self,output,label):
        output = output.strip().lower()
        label = label.strip().lower()
        label = label.replace(';',' ')
        label = label.replace(',',' ')
        label_item = label.split()
        flag = 1
        for item in label_item:
            if not(item in output):
                flag = 0
        return flag

    def get_result(self,file_path,output_path):
        all_nums = 0
        count = 0
        task = file_path.split('/')[-1]
        file_path = os.path.join(file_path,'generated_predictions.jsonl')

        with open(file_path,'r',encoding='utf-8') as f:
            contents = json.load(f)
            for content in contents:
                count+=self.preprocess_output_qa_cn(content['predict'],content['label'])
                all_nums+=1

        accuracy = count/all_nums
        with open(os.path.join(output_path,f'{task}_result.json'),'w',encoding='utf-8') as f:
            json.dump({"accuracy":f'{accuracy:.4f}'},f,ensure_ascii=False)
        return accuracy

#总结评测
class Summary_eval:
    def get_result(self,file_path,output_path):
        references = []
        candidates = []
        task = file_path.split('/')[-1]
        file_path = os.path.join(file_path,'generated_predictions.jsonl')

        with open(file_path,'r',encoding='utf-8') as f:
            contents = json.load(f)
            for content in contents:
                references.append(content['label'])
                candidates.append(content['predict'])
                
        # 计算句子级 BLEU 分数（逐条计算）
        score_dict = {"rouge-1": [], "rouge-2": [], "rouge-l": [], "bleu-4": []}
        for pred, label in zip(candidates, references):
            hypothesis = list(jieba.cut(pred))
            reference = list(jieba.cut(label))

            if len(" ".join(hypothesis).split()) == 0 or len(" ".join(reference).split()) == 0:
                result = {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}
            else:
                rouge = Rouge()
                scores = rouge.get_scores(" ".join(hypothesis), " ".join(reference))
                result = scores[0]

            for k, v in result.items():
                score_dict[k].append(round(v["f"] * 100, 4))

            bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
            score_dict["bleu-4"].append(round(bleu_score * 100, 4))


        result = {k: float(np.mean(v)) for k, v in score_dict.items()}
        with open(os.path.join(output_path,f'{task}_result.json'),'w',encoding='utf-8') as f:
            json.dump(result,f,ensure_ascii=False)
        number = 0
        for item in result.values():
            number += item/100
        return number / 4



#分类评测
class classify_cn_eval:
    def __init__(self) -> None:
        self.mydict = {'积极':'2','中性':'1','消极':'0'}
        self.stock_dict = {'下跌':'0','上涨':'1'}
        self.tasks_enable = {'stockA_prediction':self.preprocess_output_stock,
                             'finfe':self.preprocess_output_finfe_cn,
                             'Fineval-multiple_choice':self.preprocess_output_choice,
                             'CPA':self.preprocess_output_choice,
                             'CMIN-CN-Windows-test':self.preprocess_output_stock
                             }
        self.task_label_enable = {'stockA_prediction':["0","1"],
                                'finfe':["0","1","2"],
                                'Fineval-multiple_choice':["a","b","c","d"],
                                'CMIN-CN-Windows-test':["0","1"],
                                'CPA':["a","b","c","d"]
                             }


    def preprocess_output_finfe_cn(self,output):
        output = output.strip().lower()
        answer_match_cn = re.search(r'(积极|中性|消极)',output)
        answer_match = re.search(r'(0|1|2)',output)

        if answer_match_cn:
            temp = answer_match_cn.group(1)
            temp = self.mydict[temp]
        elif answer_match:
            temp = answer_match.group(1)
        else:
            temp = "unknown"
        return temp

    def preprocess_output_stock(self,output):
        output = output.strip().lower()
        answer_match_cn = re.search(r'(下跌|上涨)',output)
        answer_match = re.search(r'(0|1)',output)

        if answer_match_cn:
            temp = answer_match_cn.group(1)
            temp = self.stock_dict[temp]
        elif answer_match:
            temp = answer_match.group(1)
        else:
            temp = "unknown"
        return temp

    def preprocess_output_choice(self,output):
        output = output.strip().lower()
        answer_match = re.search(r'(a|b|c|d)',output)

        
        if answer_match:
            temp = answer_match.group(1)
        else:
            temp = "unknown"
        return temp

    def get_result(self,file_path,output_path):
        pred = []
        label = []
        task = file_path.split('/')[-1]
        file_path = os.path.join(file_path,'generated_predictions.jsonl')
        preprocess_output = self.tasks_enable[task]
        
        with open(file_path,'r',encoding='utf-8') as f:
            contents = json.load(f)
            for content in contents:
                pred_temp = preprocess_output(content['predict'])
                label_temp = preprocess_output(content['label'])

                if pred_temp == "unknown":
                    label_dict = self.task_label_enable[task][:]
                    for i in range(len(label_dict)):
                        if label_temp == label_dict[i]:
                            label_dict.pop(i)
                            break
                    pred_temp = random.choice(label_dict)

                pred.append(pred_temp)
                label.append(label_temp)
                

        
        accuracy = accuracy_score(label, pred)
        precision = precision_score(label, pred, average="weighted")
        recall = recall_score(label, pred, average="weighted",zero_division=0)
        f1 = f1_score(label, pred, average="weighted")
        mcc = matthews_corrcoef(label, pred)
        result = {
            'accuracy':accuracy,
            'precision':precision,
            'recall':recall,
            'weighted-F1':f1,
            'MCC':mcc
        }
        with open(os.path.join(output_path,f'{task}_result.json'),'w',encoding='utf-8') as f:
            json.dump(result,f,ensure_ascii=False)
        if task == 'stockA_prediction':
            return (f1+mcc) / 2
        else:
            return (f1+accuracy) /2

Model = 'touchstone_sft_v2' #模型名字

eval_dataset_path = "/finance_ML/FinLLM/FinLLM-benchmark/dataset_process" #测评工作空间地址 llamafactory形式

output_dir = f"/finance_ML/FinLLM/eval/{Model}"


if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Folder '{output_dir}' created successfully.")
else:
    print(f"Folder '{output_dir}' already exists.")


class_enabel = {
    'stockA_prediction':classify_cn_eval,
    'finfe':classify_cn_eval,
    'Fineval-multiple_choice':classify_cn_eval,
    'CPA':classify_cn_eval,
    'finna':Summary_eval,
    'finre':QA_cn_eval,
    'finqa':QA_cn_eval,
    'finnl':QA_cn_eval,
    'finese':QA_cn_eval,
    'fincqa':QA_cn_eval,
    'CMIN-CN-Windows-test':classify_cn_eval, 


    'flare-edtsum':Summary_eval,
    'flare-fpb':classify_en_eval,
    'fiqasa':classify_en_eval,
    'flare-headlines':classify_en_eval,
    'flare-fomc':classify_en_eval,
    'CFA-multiple_choice':classify_en_eval,
    'DJIA_stock_prediction':classify_en_eval,
    'cra-lendingclub':classify_en_eval,
    'flare-ner':Ner_en_eval,
    'flare-convfinqa':QA_en_eval,
    'flare-finqa':QA_en_eval,
    'CMIN-US-Windows-test':classify_en_eval,
    'fingpt-finred':RE_en_eval                  
}
count = 0
result = 0
for folder in tqdm(os.listdir(eval_dataset_path)):
    #判断是否为文件夹，如果是就去拿llama-factory的generated_predictions.jsonl
    dir_path = os.path.join(eval_dataset_path,folder)
    if os.path.isdir(dir_path) and folder in class_enabel.keys():
        classchoice = class_enabel[folder]
        evalclass = classchoice()
        result += evalclass.get_result(dir_path,output_dir)
        count += 1
print(f'average-result:{result/count:.4f}')

        
        
    






