import numpy as np
import SimpleITK as sitk
import os
import argparse

parser = argparse.ArgumentParser(description='Resampling CT img or seg to 256 or 512')
parser.add_argument('-p1', '--path1_filename', default=None, type=str,
                    metavar='path1_filename',
                    help='Raw file folder ')

parser.add_argument('-p2', '--path2_filename', default=None, type=str,
                    metavar='path2_filename',
                    help='New file folder ')
# parser.add_argument('-n', '--name', default=None, type=str,
#                     metavar='name',
#                     help='Name of folder that contains all file')
parser.add_argument('-s1', '--size1', default=256, type=int,
                    metavar='size1',
                    help='row size after resampling')
parser.add_argument('-s2', '--size2', default=256, type=int,
                    metavar='size2',
                    help='col size after resampling')

def ResampleBySize_view(path1,path2,name,rowSize,colSize):
    img = sitk.ReadImage(path1+name)
#    myshow(img,'1')
#    pix_resampled = (sitk.GetArrayFromImage(img).astype(dtype=float))
#    plot_3d(pix_resampled,0)

    original_spacing = img.GetSpacing()
    
    print('original_spacing:',original_spacing)

    original_size = img.GetSize()
    print('original_size:',original_size)
  
    #VolSize = original_size
    rowSize,colSize=rowSize,colSize
    
    factor3=1
    
    factor1=rowSize/img.GetSize()[0]
    
    factor2=colSize/img.GetSize()[1]
   
    factor=[factor1,factor2,factor3]
    
    
    
    #we rotate the image according to its transformation using the direction and according to the final spacing we want
 
    newSize = np.asarray(img.GetSize()) * factor + 0.00001
    
    dstRes =np.asarray(img.GetSpacing())/factor

    print(dstRes)
    print(newSize)
#    ret = np.zeros([newSize[0],newSize[1], newSize[2]], dtype=np.float32)
    
    newSize = newSize.astype(dtype=int).tolist()
    print(newSize)
 
    T = sitk.AffineTransform(3)
    #T.SetMatrix(img.GetDirection())
#    T.Scale(factor)
#    T.scale(factor)
 
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(img)
    resampler.SetOutputSpacing([dstRes[0], dstRes[1], dstRes[2]])
    resampler.SetSize(newSize)
    method=sitk.sitkLinear
    resampler.SetInterpolator(method)
    resampler.SetTransform(T)

    imgResampled = resampler.Execute(img)
 
    new_spacing = imgResampled.GetSpacing()
    
    print('new_spacing:', new_spacing)
    print('new_size:', imgResampled.GetSize())
    path2=path2
    sitk.WriteImage(imgResampled,path2+name) 

if __name__ == "__main__":
    global args
    args = parser.parse_args()
    p1 = args.path1_filename
    p2 = args.path2_filename
    s1 = args.size1
    s2 = args.size2

    for file in os.listdir(p1):
        ResampleBySize_view(p1, p2, file, s1, s2)
    

      
        
    