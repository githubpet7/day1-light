from PIL import Image as Image
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("====== PIL Image =====")
    # read PIL image
    pil_image = Image.open("./1.png").convert("RGB")
    print("The size of the pil image: ",pil_image.size)
    print("The mode of the pil image: ",pil_image.mode)
    print("The type of the pil image: ",type(pil_image))
    
    # pil image to np array
    pil_image_array = np.array(pil_image)
    print("Array shape of pil image: ",pil_image_array.shape) # H W C   >>> RGB
    
    # read cv2 image
    print("===== CV2 =====")
    cv_image = cv2.imread("./1.png")
    print("The shape of CV2 image (Before crop):", cv_image.shape)
    # crop your image 
    cv_image = cv_image[20:20+300,10:10+400] # h ,w
    print("The shape of CV2 image (After crop):", cv_image.shape)

    # split your image along channel and save b-channel
    (cv2_image_b, cv2_image_g, cv2_image_r) = cv2.split(cv_image)
    cv2.imwrite("./cv2_image_b_channel.png", cv2_image_b.astype(np.uint8))  # uint8 uint16 are acceptable
    
    # convert your image for BGR to Grayscale
    cv_image = cv2.cvtColor(cv_image,cv2.COLOR_BGR2GRAY)
    
    # show your cv2 image
    print("The shape of cv2 image(Gray Scale): ",cv_image.shape) # H W C >>>>> Gray Scale
    cv2.imshow("cv_image", cv_image)
    cv2.waitKey()
    
    # save your image
    cv2.imwrite("./cv_1.png", cv_image.astype(np.uint8))  # uint8 uint16 are acceptable
    
    print("===== Tensor Operate ======")
    
    print("The Shape of cv2 image: ",cv_image.shape)
    cv_image_tensor = F.to_tensor(cv_image) # H W C > C H W
    print("After <to_tensor> operate, Tensor shape is:", cv_image_tensor.shape) 
    cv_image_tensor = cv_image_tensor.permute(1,2,0) # C H W > H W C
    print("After <permute> operate, Tensor shape is",cv_image_tensor.shape)

    
    # move our tnesor to GPU and back to cpu
    cv_image_tensor = cv_image_tensor.to(device)
    print("After <to device> operate, Show where the tensor is: ",cv_image_tensor.device)
    cv_image_tensor = cv_image_tensor.cpu()
    print("After <cpu()> operate, Show where the tensor is: ",cv_image_tensor.device)
    
    # convert our tensor to numpy array
    print("Show the type of tensor: ",type(cv_image_tensor))
    cv_image = cv_image_tensor.numpy() # this will work only when our tensor on cpu
    print("After <numpy()> operate, Show where the cv_image is: ",type(cv_image))

