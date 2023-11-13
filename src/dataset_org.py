import os
import torch
import torchvision.transforms as transforms
from transformers import BertTokenizer, AutoTokenizer
import pickle
import pandas as pd
import numpy as np
from PIL import Image
import io, base64
from io import BytesIO

import pickle5 as pickle
import fasttext
from torch.utils.data import Dataset

import tqdm
import random

env2clusterpath = {'triton':'/scratch/work/nasibun1/projects/XMC/multimodalxmc/data/MMAmazonTitles-300K',
            'puhti':'/scratch/project_2001083/nasib/XMC/multimodalxmc/data/MMAmazonTitles-300K',
            'mahti':'/scratch/project_2001083/nasib/XMC/multimodalxmc/data/MMAmazonTitles-300K'}


def collate_fn(batch): 
    '''
    will only work for multimodal settings
    '''
    input_ids =[item[0][0] for item in batch]
    input_ids = torch.stack(input_ids)
    attention_mask = [item[0][1] for item in batch]
    attention_mask = torch.stack(attention_mask)
    token_type_ids = [item[0][2] for item in batch]
    token_type_ids = torch.stack(token_type_ids)
    visual_embeds = [item[0][3] for item in batch]
    visual_embeds = torch.stack(visual_embeds)
    visual_attention_mask = [item[0][4] for item in batch]
    visual_attention_mask = torch.stack(visual_attention_mask)
    visual_token_type_ids = [item[0][5] for item in batch]
    visual_token_type_ids = torch.stack(visual_token_type_ids)
    
    label_ids = [item[1] for item in batch]
    label_ids = torch.stack(label_ids)
    group_label_ids = [item[2] for item in batch]
    group_label_ids = torch.stack(group_label_ids)
    candidates = [item[3] for item in batch]
    candidates = torch.stack(candidates)
    
    return input_ids,attention_mask,token_type_ids,visual_embeds,visual_attention_mask,visual_token_type_ids,label_ids,group_label_ids,candidates
    
    
class DataHandler:
    '''
    Handle all the data reading, preprocessing ,dataset, dataloader and other stuff.
    Images in Train set: min:1, max:50, avg: 4.911
               Test set: min:1,  max:50, avg: 4.85
    
    '''
    def __init__(self,cfg,path):
        #check_path_existance(path)
        self.cfg = cfg
        self.path = path
        self.label_map = {}
        self.read_files(cfg,path)
        self.group_y = None
        if cfg.use_meta:
            self.group_y = self.load_group()
        
        #print(self.group_y)
        

    
    def read_files(self,cfg,path):
        #reading text files
        train_text, train_uids = self.readfile(path.train_text_file)
        test_text, test_uids = self.readfile(path.test_text_file)
        label_text, label_uids = self.readfile(path.label_text_file)
        
        self.train_text_dict = dict(zip(train_uids,train_text))
        self.test_text_dict = dict(zip(test_uids,test_text))
        self.label_text_dict = dict(zip(label_uids,label_text))
        
        #reading ground truth
        train_labels = self.read_label_files(path.train_X_Y_file)
        test_labels = self.read_label_files(path.test_X_Y_file)
        
        self.train_label_dict = dict(zip(train_uids,train_labels))
        self.test_label_dict = dict(zip(test_uids,test_labels))
        self.label2uid_dict = dict(zip(range(len(label_uids)),label_uids)) #what use? label inex to label uid

        #reading image files
        self.train_img_uids = self.read_uids(path.train_img_uid_file)
        self.test_img_uids = self.read_uids(path.test_img_uid_file)
        self.label_img_uids = self.read_uids(path.label_img_uid_file)
    
    
    def read_uids(self,filepath):
        uids = []
        f = open(filepath,'r')
        for line in f.readlines():
            uids.append(line.strip())
        return uids

    def readfile(self,filepath):
        data,label = [], []
        f = open(filepath,'r')
        for line in f.readlines():
            data.append(line.split('->')[1].strip())
            label.append(line.split('->')[0])
            
        return data,label
    
    def load_group(self):
        #need to add this path to config
        if self.cfg.dataset == 'mmamazontitles300k':
            cluster_path = env2clusterpath[self.cfg.cluster] + f'/label_group{self.cfg.group_y_group}.npy'
            print('cluster path:',cluster_path)
            groups =  np.load(cluster_path, allow_pickle=True)
            for i in range(len(groups)):
                for j in range(len(groups[i])):
                    groups[i][j] = int(groups[i][j] )
                    
            return groups
            
    
    def read_label_files(self,filename):
        labels = []
        f = open(filename,'r')
        for line in f.readlines()[1:]:
            labels.append([int(x.split(':')[0]) for x in line.strip().split()])
        return labels
    
    def getDatasets(self):
        
        train_dset = MMDataset(self.cfg,self.path,'train',self.train_text_dict,self.train_label_dict,
                               self.label_text_dict,self.label2uid_dict,self.group_y)
        test_dset = MMDataset(self.cfg,self.path,'test',self.test_text_dict,self.test_label_dict,
                               self.label_text_dict,self.label2uid_dict,self.group_y)
        
        return train_dset,test_dset
        #return test_dset

class MMDataset(Dataset):
    def __init__(self,cfg,path,mode,text_dict,label_dict,label_text_dict,label2uid_dict,group_y=None,candidates_num=None):
        
        '''
        Multimodal Dataset.
        Args:
            cfg: configuration object. consists of all hyperparameter and other settings.
            path: path object consists of oaths to all files related to the corrsponding dataset specified in cfg object.
            mode: train or test. currently doesn't use valid mode.
            text_dict: 
            label_dict:
            label_text_dict:
            label2uid_dict:
            group_y:
            
        Return:
            data:
            label_ids:
            group_label_ids:
            candidates:

        '''
        
        self.cfg = cfg
        self.path = path
        self.mode = mode
        self.text_dict = text_dict
        self.uid_list = list(text_dict.keys())
        self.label_dict = label_dict
        self.label2uid_dict = label2uid_dict
        self.n_labels = len(self.label2uid_dict)
        #label text dictionary (if use label text features)
        self.label_text_dict = label_text_dict
        self.len = len(self.uid_list )
        
        self.candidates_num = cfg.group_y_candidate_num 
        self.group_y = group_y
        self.multi_group = False
        fasttextflag = True
        if cfg.model_type=='text':
            if fasttextflag == True:
                pass
                #do something
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(cfg.encoder_model)
        if cfg.model_type=='multimodal':
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            
        self.transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])
        
        if cfg.model_type == 'image-only':
            if mode=='train':
                f = open(path.train_img_file,'rb')
                self.img_dict = pickle.load(f) 
                f.close()
            elif mode == 'test':
                f = open(path.test_img_file,'rb')
                self.img_dict = pickle.load(f) 
                f.close()
        elif cfg.model_type == 'multimodal':
            if mode=='train':
                f = open(path.train_object_features,'rb')
                self.object_features = pickle.load(f)
                f.close()
            elif mode =='test':
                f = open(path.test_object_features,'rb')
                self.object_features = pickle.load(f)    
                f.close()
        else: 
            pass 
        
        if cfg.use_label_image:
            #if use label image features
            f = open(path.label_img_file,'rb')
            #label image dictionary
            self.label_img_dict = pickle.load(f)
        self.candidates_num = candidates_num 
        label_map=label2uid_dict
        if group_y is not None:
            # group y mode

            self.candiates_num, self.group_y, self.n_group_y_labels = candidates_num, [], group_y.shape[0]
            self.map_group_y = np.empty(len(label_map), dtype=np.long)
            #print('length of map_group_y:',len(self.map_group_y))
            for idx, labels in enumerate(group_y):
                self.group_y.append([])
                for label in labels:
                    label_random = random.choice(range(self.n_group_y_labels))
                    label_idx = label_map.get(label, label_random)
                    self.group_y[-1].append(label_idx)
                    self.map_group_y[label_idx] = idx
                    #self.group_y[-1].append(label_map[label])
                #print('self.group_y[-1]:',self.group_y[-1])
                for i in range(len(self.map_group_y)):
                    val = self.map_group_y[i]
                    if val<0 or val>n_group_y_labels:
                        self.map_group_y[i] = random.choice(range(self.n_group_y_labels))
                #self.map_group_y[self.group_y[-1]] = idx #check this line
                self.group_y[-1]  = np.array(self.group_y[-1])
            self.group_y = np.array(self.group_y)
            for i in range(len(self.map_group_y)):
               val = self.map_group_y[i]   
               if val<0 or val>n_group_y_labels:
                   self.map_group_y[i] = random.choice(range(self.n_group_y_labels))
            
    def __getitem__(self,idx):
        
        uid = self.uid_list[idx]

        fasttextflag = True
        if fasttextflag == True:
            text = self.text_dict[uid].lower()
            words = text.split()
            fasttext_model = fasttext.load_model('/scratch/project_2001083/sanchit/models/crawl-300d-2M-subword.bin')
            word_vectors = [fasttext_model.get_word_vector(word) for word in words]
            input_ids = torch.tensor(word_vectors, dtype=torch.float32)
            max_length = self.cfg.text_input_length
            max_tokens_to_attend = min(len(words), max_length)
            if len(words) > max_length:
                words = words[:max_length]
            else:
                words += [""] * (max_length - len(words))
            tokenized_text = " ".join(words)
            attention_mask = [1] * max_tokens_to_attend + [0] * (max_length - max_tokens_to_attend)
            attention_mask = torch.tensor(attention_mask, dtype=torch.float32)
            token_type_ids = [0] * max_length
            token_type_ids = torch.tensor(token_type_ids, dtype=torch.float32)
            #input_ids = torch.tensor(tokenized_text, dtype=torch.float32)
            #data = [input_ids, attention_mask, token_type_ids]
            data = [input_ids]
            return data



            #do something
        else:
            if not self.cfg.model_type=='image-only':
                #text tensors
                text = self.text_dict[uid].lower()
                tokens = self.tokenizer(text, padding='max_length', max_length=self.cfg.text_input_length,truncation=True)
                input_ids = torch.tensor(tokens["input_ids"])
                attention_mask = torch.tensor(tokens["attention_mask"])
                token_type_ids = torch.tensor(tokens["token_type_ids"])
                data = [input_ids, attention_mask, token_type_ids]

        if self.cfg.model_type=='image-only':
            #Randomly choosing a single image 
            img_tensor = self.transform(Image.open(io.BytesIO(base64.decodebytes(bytes(random.choice(self.img_dict[uid]), "utf-8")))))
            data = [img_tensor]
            
        if self.cfg.model_type=='multimodal':
            visual_embeds = torch.tensor(np.stack(self.object_features[uid])).view(-1,1024)
            if len(visual_embeds)>=self.cfg.max_object_fature_number:
                visual_embeds = visual_embeds[0:self.cfg.max_object_fature_number]
            else:
                pad_number = self.cfg.max_object_fature_number-len(visual_embeds)
                visual_embeds = torch.cat((visual_embeds,torch.zeros(pad_number,1024)),0)
                #print('visual embeds size:',visual_embeds.shape)
            #visual_embeds = torch.tensor(random.choice(self.object_features[uid]))
            visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)
            visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)
            data = [input_ids, attention_mask, token_type_ids,visual_embeds,visual_attention_mask,visual_token_type_ids]
            
        
        if self.group_y is not None:  
            labels = self.label_dict[uid]
            #making group labels
            label_ids = torch.zeros(self.n_labels)
            label_ids = label_ids.scatter(0, torch.tensor(labels),torch.tensor([1.0 for i in labels]))
            group_labels = self.map_group_y[labels]
            if self.multi_group:
                group_labels = np.concatenate(group_labels)
            group_label_ids = torch.zeros(self.n_group_y_labels)
            group_label_ids = group_label_ids.scatter(0, torch.tensor(group_labels),torch.tensor([1.0 for i in group_labels]))
            
            candidates = np.concatenate(self.group_y[group_labels], axis=0)

            if len(candidates) < self.candidates_num:
                sample = np.random.randint(self.n_group_y_labels, size=self.candidates_num - len(candidates))
                candidates = np.concatenate([candidates, sample])
            elif len(candidates) > self.candidates_num:
                candidates = np.random.choice(candidates, self.candidates_num, replace=False)

            if self.mode == 'train':
                return data, label_ids[candidates], group_label_ids, candidates
            else:
                return data,label_ids, group_label_ids, candidates
        else:
            labels = self.label_dict[uid]
            #making group labels
            label_ids = torch.zeros(self.n_labels)
            label_ids = label_ids.scatter(0, torch.tensor(labels),torch.tensor([1.0 for i in labels]))
            return data, label_ids
            
        
        
    def __len__(self):
        return self.len 
