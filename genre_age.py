import cv2
#le modèle AgeNet:
model_mean_value= (78.4463377603, 
                   87.7689143744, 
                   114.895847746)

#liste pour l'age :
age_list=['(0,2)','(4,6)','(8,12)',
          '(15,20)','(25,32)','(38,43)',
          '(48,53)','(60,100)']

#list pour detecter le sex 
genre_list=['MALE','FEMALE']

def files_get():
    age_net=cv2.dnn.readNetFromCaffe(
          'file_age_sex_just\\age_deploy.prototxt',
          'file_age_sex_just\\age_net.caffemodel'
    )
    genre_net=cv2.dnn.readNetFromCaffe(
        'file_age_sex_just\\gender_deploy.prototxt',
        'file_age_sex_just\\gender_net.caffemodel'
    )
    return(age_net,genre_net)


def read_image(age_net,genre_net):
    
    #choisir type d'ecriture
    font=cv2.FONT_HERSHEY_SIMPLEX
    image=cv2.imread('photo\\man-657869_1280.jpg')
    
    #fichier pour detecter le visage :
    
    
    face_cascade=cv2.CascadeClassifier('file_age_sex_just\haarcascade_frontalface_alt.xml')

    
    #convertit une image couleur en une image en niveaux de gris
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    
    #détecte les visages dans une image en niveaux de gris
   
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if (len(faces)>0):
        print("found {} faces".format(str(len(faces))))
        
    for (x,y,w,h) in faces:
        
        #designe un carre:
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,255,0),2)
        
        #prendre un copier de face et donner au algorithme:
        face_img=image[y:y+h,x:x+w].copy()
        blob = cv2.dnn.blobFromImage(face_img, scalefactor=1.0, size=(227, 227), mean=(78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
        
        #detecter le sex:
        genre_net.setInput(blob)
        gendre_p=genre_net.forward()
        genre=genre_list[gendre_p[0].argmax()]
        
        print("genre:"+ genre)
        
        
        #detecter l'age:
        age_net.setInput(blob)
        age_p=age_net.forward()
        age=age_list[age_p[0].argmax()]
        
        print("genre:"+ age)
        G_A="%s %s" % (genre,age)
        cv2.putText(image, G_A, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

       
        cv2.imshow('amira',image)
     # Attendre l'appui sur la touche 'q' pour fermer la fenêtre
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    cv2.destroyAllWindows()

# Fermer toutes les fenêtres
cv2.destroyAllWindows()
if __name__=="__main__" :
    age_net,genre_net=files_get()
    read_image(age_net,genre_net)  