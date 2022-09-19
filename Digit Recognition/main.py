'''
Source for display draw screen: https://github.com/banipreetr/Real-Time-Digit-Recognition
'''
import pygame
import time
import cv2
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from model import LeNet5
import matplotlib.pyplot as plt

## Load model
num_classes = 10
device = 'cuda' if torch.cuda.is_available else 'cpu'

model = LeNet5(num_classes).to(device)
state = torch.load('D:\LapTrinh\Data Science\ComputerVision\Digit Recognition\checkpoint\model.pt')
model.load_state_dict(state['state_dict'])
model.eval()

## preprocessing image from transform
transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)) # normalize data
])

## Create the screen to draw a digit
pygame.init()

display_width = 300
display_height = 300
radius = 12

black = (0,0,0) # RGB
white = (255,255,255)
img = np.array(0)

gameDisplay = pygame.display.set_mode((display_width,display_height))
pygame.display.set_caption('Drawing pad')

def textObjects(text, font):
    textSurface = font.render(text, True, white)
    return textSurface, textSurface.get_rect()

def message_display(text, locx, locy,size):
    largeText = pygame.font.Font('freesansbold.ttf', size) # Font(font name, font size)
    TextSurf, TextRec = textObjects(text, largeText)
    TextRec.center = (locx,locy)
    gameDisplay.blit(TextSurf, TextRec)
    pygame.display.update()
def gameLoop():
    gameExit = False
    gameDisplay.fill(black)
    pygame.display.flip()
    tick = 0
    tock = 0
    startDraw = False
    while not gameExit:
        if tock - tick >= 2 and startDraw:
            data = pygame.image.tostring(gameDisplay, 'RGBA')
            img = Image.frombytes('RGBA', (display_width, display_height), data)
            img = img.convert('L')
            img = transform(img)
            
            predVal = model(img.unsqueeze(0).to(device)).argmax(dim = 1).item()
            
            gameDisplay.fill(black)
            message_display("Predicted Value: "+str(predVal), int(display_width/2), int(display_height/2), 20)
            time.sleep(2) #sleep for 2 seconds
            gameDisplay.fill(black)
            pygame.display.flip()
            tick = 0
            tock = 0
            startDraw = False
            continue
        
        tock = time.time()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                gameExit = True
        if pygame.mouse.get_pressed()[0]:
            spot = pygame.mouse.get_pos()
            pygame.draw.circle(gameDisplay,white,spot,radius)
            pygame.display.flip()
            tick = time.time()
            startDraw = True
gameLoop()
pygame.quit()