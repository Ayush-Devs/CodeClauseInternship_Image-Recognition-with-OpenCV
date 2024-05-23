import cv2
import os

def detect_faces(image_path, face_cascade):
    
    img = cv2.imread(image_path)
    '''
    if img is None:
        print("Failed to load image:", image_path)
        return None  
        '''

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    if len(faces) > 0:
        print("Faces detected in", image_path, ":", len(faces))
    else:
        print("No faces detected in", image_path)

    return img

def main():
    cascade_path = 'haarcascade_frontalface_default.xml'

    if not os.path.exists(cascade_path):
        print("Error: Haar Cascade file not found at", cascade_path)
        return

    face_cascade = cv2.CascadeClassifier(cascade_path)

    directory = 'img'

    if not os.path.exists(directory):
        print("Error: Directory not found at", directory)
        return

    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(directory, filename)
            img = detect_faces(image_path, face_cascade)
            if img is not None:
                cv2.imshow(filename, img)

    print("Press any key to close the windows...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
