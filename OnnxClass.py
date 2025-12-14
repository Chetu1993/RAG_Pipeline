from transformers import AutoTokenizer
import numpy as np
import onnxruntime as ort
from langchain.embeddings.base import Embeddings
from huggingface_hub import hf_hub_download


class OnnxMiniEmbeddings(Embeddings):
    def __init__(self):
        self.model_path=hf_hub_download(repo_id='Xenova/all-MiniLM-L6-v2',filename='onnx/model.onnx')
        self.max_length=128
        self.session=ort.InferenceSession(self.model_path,provider=['CPUExecutionProvider'])

        self.tokenizer=AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

    def provider_input(self,text):
        token=self.tokenizer(text,truncation=True,padding="max_length",return_tensors="np",max_length=self.max_length)
        if "token_type_ids" not in token:
            token['token_type_ids']=np.zeros_like(token['input_ids'])

        return {'token_type_ids':token['token_type_ids'],
                "attention_mask":token['attention_mask'],
                "input_ids":token["input_ids"]}

    def mini_pool(self,token_embeddings,attention_mask):
        mask_expand=attention_mask[...,None]

        pool=(token_embeddings*mask_expand).sum(1)/mask_expand.sum(1)
        return pool


    def embed_query(self,text):
        inputs=self.provider_input(text)
        outputs=self.session.run(None,inputs)
        sentence_embeddings=self.mini_pool(outputs[0],inputs['attention_mask'])
        return sentence_embeddings[0].tolist()

    def embed_documents(self,texts):
        vectors=[]
        for t in texts:
            inputs=self.provider_input(t)
            outputs=self.session.run(None,inputs)
            sentence_embeddings=self.mini_pool(outputs[0],inputs["attention_mask"])
            vectors.append(sentence_embeddings[0].tolist())
        return vectors

