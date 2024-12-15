from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2

class FaceDetection:
    def __init__(self):
        # ydetecti el blasa mta3 lwejh
        self.mtcnn = MTCNN(image_size=160, margin=0)

        # ybadel elwejh vector
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval()

    def getFace(self, img):
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        # Get cropped and prewhitened image tensor
        img_cropped = self.mtcnn(img) #ya3tik elwejh mta3 el image "cropped image"
        # cv2.imshow(img_cropped.permute(1,2,0).numpy() * 255) #n7awlouh l7aja yefhemha cv2
        return img_cropped
    
    def getEmbedding(self, img):
        img_embedding = self.resnet(img.unsqueeze(0))
        return img_embedding.squeeze(0).detach().cpu().numpy()