from fastapi import FastAPI
import sys
import os

# Add src directory to sys.path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pose_identifier.pose_identifier as pose_identifier

PORT: int = 7272

# initiate app
app = FastAPI()


@app.get("/")
def root():
    # pose_identifier should identify new pose in real time?
    # pose = pose_identifier # someone explain to me what do i need to read to get the pose
    personthere: bool = True # can i get this as well somehow?
    return {"pose": "standing", "person_there": personthere}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT) # localhost for now