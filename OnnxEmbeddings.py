from huggingface_hub import hf_hub_download
import onnxruntime as ort
import numpy as np
from langchain.embeddings.base import Embeddings
from transformers import AutoTokenizer

class OnnxClass(Embeddings):
    def __init__(self):
        self.model=hf_hub_download(repo_id='Xenova/all-MiniLM-L6-v2',filename="onnx/model.onnx")
        self.tokenizer=AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.session=ort.InferenceSession(self.model,providers=['CPUExecutionProvider'])
        self.max_length=128


    def provider_input(self,text):
        token=self.tokenizer(text,padding='max_length',truncation=True,max_length=self.max_length,return_tensors="np")

        if 'token_type_ids' not in token:
            token['token_type_ids']=np.zeros_like(token['input_ids'])

        return {'input_ids':token['input_ids'],
                'attention_mask':token['attention_mask'],
                'token_type_ids':token['token_type_ids']}

    def pooling(self,token_embeddings,attention_mask):
        mask_expand=attention_mask[...,None]
        pooled=(token_embeddings*mask_expand).sum(1)/mask_expand.sum(1)

        return pooled

    def embed_query(self,text):
        inputs=self.provider_input(text)
        ouputs=self.session.run(None,inputs)

        embedding_documents=self.pooling(ouputs[0],inputs['attention_mask'])
        return embedding_documents[0].tolist()

    def embed_documents(self,texts):
        vectors=[]
        for text in texts:
            inputs=self.provider_input(text)
            outputs=self.session.run(None,inputs)
            embedding_documents=self.pooling(outputs[0],inputs['attention_mask'])
            vectors.append(embedding_documents[0].tolist())

        return vectors

