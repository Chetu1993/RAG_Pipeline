from fastapi import FastAPI,File,UploadFile
import shutil
import os
import uvicorn
data_dir=('/data/upload')
os.makedirs(data_dir,exist_ok=True)

app=FastAPI()

@app.post("/upload")
async def upload_file(file:UploadFile = File(...)):
    file_path=os.path.join(data_dir,file.filename)
    with open(file_path,"wb") as buffer:
        shutil.copyfileobj(file.file,buffer)

    return {"status":"saved","path":file_path}


if __name__=="__main__":
    uvicorn.run("getting_image_file:app",host="0.0.0.0",port=8000)







