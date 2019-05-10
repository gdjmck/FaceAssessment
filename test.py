import torchsummary
import model

face_assess = model.FaceAssess()

if __name__ == '__main__':
    torchsummary.summary(face_assess, (3, 1024, 1024))