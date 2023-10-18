from ultralytics import YOLO
model=YOLO('face_yolov8n.pt')
from keras_facenet import FaceNet
embedder_facenet = FaceNet()
import joblib
import cv2
import numpy as np


# Load the gender prediction model
gender_clf = joblib.load('GENDER_knn_model.pkl')

# Load the age prediction model
age_regressor = joblib.load('AGE_knn_model.pkl')

# Load the ethnicity prediction model
ethnicity_clf = joblib.load('ETHNICITY_knn_model.pkl')


def detect_face(frame, model):
    
    """
    Detects the face in the given frame using YOLO.
    """        
    results = model(frame)
    #print(results)
    #test=results[0].orig_img
    #print(test)
    # Assuming that we consider the first detected face
    faces_arrays = results[0].boxes.xyxy.numpy().astype(int)
    if faces_arrays.shape[0]>0:
        
        return faces_arrays

      
    else:
        print("No faces detected.")
        return None

def image_to_embedding(face):
    face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
    expanded_img_array = np.expand_dims(face, axis=0)
    embedding = embedder_facenet.embeddings(expanded_img_array)
    return embedding

def predict_using_embedding(embedding, gender_model, age_model, ethnicity_model):
    """
    Use the given embedding to predict gender and age.
    """
    # Make predictions using the embedding and models
    gender = gender_model.predict(embedding)
    age= age_model.predict(embedding)
    ethnicity = ethnicity_clf.predict(embedding)
    
    gender = "Male" if gender[0] == 0 else "Female"
    # Map ethnicity prediction integer to label
    ethnicity_labels = {
        0: 'White',
        1: 'Black',
        2: 'Asian',
        3: 'Indian',
        4: 'Others'
    }
    ethnicity = ethnicity_labels.get(ethnicity[0], 'Unknown')

    return gender, age, ethnicity
def process_frame(frame):
    
    frame=cv2.resize(frame, (1048, 720))

    faces = detect_face(frame, model)

    if faces is not None:
        for face in faces:
            x1, y1, x2, y2=face
            face_frame = frame[y1:y2, x1:x2]
            # Convert the face image to its embedding
            embedding = image_to_embedding(face_frame)
            # Make predictions
            gender, age, ethnicity  = predict_using_embedding(embedding, gender_clf, age_regressor, ethnicity_clf)
            x1, y1, x2, y2=face
            # Display results
            cv2.putText(frame, f"{gender}, {int(float(age))}, {ethnicity}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return frame
if __name__=='__main__':
    frame=cv2.imread('man_904.jpg')
    frame=process_frame(frame)
    # Using cv2.imshow() method 
    # Displaying the image 
    cv2.imshow('frame', frame) 

    # waits for user to press any key 
    # (this is necessary to avoid Python kernel form crashing) 
    cv2.waitKey(0) 

    # closing all open windows 
    cv2.destroyAllWindows()
