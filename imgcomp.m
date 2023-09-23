#Coded by N Gokul
#This code can be used to calculate the similarity of two images 
#This code has one limitation with the current version
#If two images has identical size to each other or one image fits inside the other then this code will run
#For instance is image1 has size 134x 100 and
#image2 has size 100 x 200, the script wont run(if image2 was 200x100 the code will run)
 
global y_n = 1;#will be used later
global raw = []; #will be used later
[imname1, impath1, fltidx1] = uigetfile({'*.png;*.jpg', 'Supported Picture Formats'},'select image 1');#getting filename and directory from user
im1 = strcat(impath1,imname1);
img1  = imread(im1);
if imfinfo(im1).BitDepth == 16#converting to appropriate format to run the code  
  img1 = uint8(img1);
endif
if imfinfo(im1).BitDepth == 1#converting to appropriate format to run the code
  img1 = (img1.*255);
endif
[imname2, impath2, fltidx2] = uigetfile({'*.png;*.jpg', 'Supported Picture Formats'}, 'select image 2');#getting filename and directory from user for image 2
im2 = strcat(impath2,imname2);
img2  = imread(im2);
if imfinfo(im2).BitDepth == 16#converting to appropriate format to run the code
  img2 = uint8(img1);
endif
if imfinfo(im2).BitDepth == 1#converting to appropriate format to run the code
  img2 = (img2.*255);
endif

function [channel_1 channel_2 channel_3]= channelise(img)
  #Used to separate coloured images to its respective component channels
  #For an RGB image channel_1 returns red cmponent, 
  #channel_l returns green component and channel_3 gives green component
  #For HSV it follows the same order: channel_1 : hue channel_2: saturation
  #channel_3 : value
  #USAGE:
  #      img = imread("image.png");
  #      [R G B] = channelise(img);
  if size(img,3)> 1
    channel_1 = img(:,:,1);
    channel_2 = img(:,:,2);
    channel_3 = img(:,:,3);   
  endif
endfunction


function [sz_1, sz_2, main_img, sec_img] = selector(img1,img2)
  #This function returns the image with larger resolution as main_img and the 
  #sec_img is the image with lower dimention(or resolution) sz_1 is the size of larger image
  #sz_2 is the sixe of smaller image 
  sz_1 = [rows(img1) columns(img1)];
  sz_2 = [rows(img2) columns(img2)];
  if sz_1(1) <= sz_2(1) && sz_1(2) <= sz_2(2);
    main_img = img2;
    sec_img = img1;
  elseif sz_1(1) >= sz_2(1) && sz_1(2) >= sz_2(2);
    main_img = img1;
    sec_img  = img2;
  endif  
endfunction 

function retval = mse(img1,img2)
  #USAGE:
  #     MSE = mse(img1,img2)
  #gives the mean squared error between two arrays img1 and img2  
  dif = sum((img1-img2).**2);
  dif = dif/(rows(img1)*columns(img1));
  retval = dif;
endfunction

function ssim = eqn_ssim(img1,img2,c1,c2)
  #Computes the ssim between two 1D arrays
  #usage:
  #     ssim = eqn_ssim(X,Y,c1,c2)
  # c1 and c2 are user defined values
  #referance :https://en.wikipedia.org/wiki/Structural_similarity#Algorithm
  s1 = std(img1,1);
  s2 = std(img2,1);
  u1 = mean(img1);
  u2 = mean(img2);
  c12 = cov(img1,img2,1);
  numer = (2*u1*u2+c1)*(c12 + c2);
  denom = (u1^2 +u2^2 +c1)*(s1^2+s2^2+c2);
  ssim = numer/denom;
endfunction

function ssim_in = ssi(img1,img2)
  #Function to calculate ssi of two images of same size
  #The constants involved in ssi are predefiend in this function   
  #If coloured image is given then the SSI of the different channels are returned as a matrix_type
  #The order of SSI of each channel is same as the order of channel in the color system
  #If a grayscale image is given the function returns the SSI as a number
  
  L = 2^8 -1;
  c1 = (.01*L)^2;
  c2 = (.03*L)^2;  
  if size(img1,3) == 3 && size(img2,3) == 3
    [r1 g1 b1]= channelise(img1);
    [r2 g2 b2]= channelise(img2);
    ch1 = eqn_ssim(r1(:),r2(:),c1,c2);
    ch2 = eqn_ssim(g1(:),g2(:),c1,c2);
    ch3 = eqn_ssim(b1(:),b2(:),c1,c2);
    ssim_in = [ch1 ch2 ch3];
  else
    ssim_in = eqn_ssim(img1(:),img2(:),c1,c2);
  endif   
endfunction

function retval = compare_process(img1,img2)
  #Function used to compare two images 
  #This function takes care of most of the job and besides returning the values it actually displays them
  #The retval can eitherbe a number or 1d array or 2d matrix depending on image properties
  #If both of the images are coloured then retval is 2D array first row giving MSE of each channels
  #and second row gives the SSI of each channels
  #If one image is grayscale then the retval has two entries first one is for mse and the second is ssi
  #If one image is smaller than the other then the above format is also applicable with exceptions of few cases
  #The vlaues returned are the most fit ssi and mse
  #If the cropped image fits at more than one places the max SSI is returned and the coordinates along with their 
  #correspondin mse and ssi are generated as a matrix 'raw'
  if size(img1,1) == size(img2,1) && size(img1,2) == size(img2,2)#condition to check the dimension of the image 
    if size(img2,3) > 1 && size(img1,3) > 1
      hsv1 = rgb2hsv(img1);#converting  rgb to hsv
      [h1 s1 v1] = channelise(hsv1);#separating the hsv channel
      hsv2 = rgb2hsv(img2);
      [h2 s2 v2] = channelise(hsv2);
      h_mse = mse(h1(:),h2(:));#mse of hue
      s_mse = mse(s1(:),s2(:));#mse of saturation
      v_mse = mse(v1(:),v2(:));#mse of value
      hsvmssi = ssi(hsv1, hsv2);#finding ssi of all channels
      printf('\n')
      printf('MSE of \nHue:%f\nsaturation :%f\nValue :%f\n', h_mse,s_mse,v_mse)
      printf('\n')
      printf('SSI of \nHue:%f\nSaturation :%f\nValue :%f\n',hsvmssi(1),hsvmssi(2),hsvmssi(3))
      printf('\n')    
      if h_mse > .998 || s_mse >.9999
        disp('The images are same')
        printf('\n')
      endif
      retval = [h_mse, s_mse, v_mse;hsvmssi(1),hsvmssi(2),hsvmssi(3)];
      return
    elseif size(img1,3)==1 && size(img2,3) ==1
      i_mse = mse(img1(:), img2(:));
      i_ssi = ssi(img1,img2);
      printf('\n')
      printf('For two images\n\nMSE :%f\nSSI :%f\n',i_mse,i_ssi)
      printf('\n')
      if i_mse < .04 || i_ssi >.999      
        disp('The images are same')
        printf('\n')
      endif
      retval = [i_mse i_ssi]
      return
    elseif size(img1,3) ==1 && size(img2,3) !=1
      img2 = rgb2gray(img2);
      i_mse = mse(img1(:), img2(:));
      i_ssi = ssi(img1(:),img2(:));
      printf('\n')
      printf('For two images\n\nMSE :%f\nSSI :%f\n',i_mse,i_ssi)
      printf('\n')
      if i_mse < .04 || i_ssi >.999
        disp('The images are same')
        printf('\n')
      endif
      retval = [i_mse i_ssi]
      return
    elseif size(img1,3) !=1 && size(img2,3) ==1
      img1 = rgb2gray(img1);
      i_mse = mse(img1(:), img2(:));
      i_ssi = ssi(img1(:),img2(:));
      printf('\n')
      printf('For two images\n\nMSE :%f\nSSI :%f\n',i_mse,i_ssi)
      printf('\n')
      if i_mse < .04 || i_ssi >.999
        disp('The images are same')
        printf('\n')
      endif
      retval = [i_mse i_ssi]
      return
    endif
    
  else
    disp('The two picture has dissimilar size. If one is a cropped image of the another then the code will work without error but takes a lot of time depending on the image size.')
    conf = yes_or_no('Do you want to continue');
    printf('\n')
    if conf == 1
      if size(img2,3) > 1 && size(img1,3) > 1    
        [sz_1, sz_2, A2, B2] = selector(img1,img2);#selecting the larger image
        A1 = rgb2hsv(A2);
        B1 = rgb2hsv(B2);
        A = A1(:,:,1);#selecting the hue channel as it is more sensible
        B = B1(:,:,1);
        corr_map = zeros([size(A,1),size(A,2)]);
        f = waitbar(0,'1','Name','finding template...',...
            'createcancelbtn','setappdata(gcbf,''canceling'',1)');#setting the progress bar
        
        for i = 1:size(A,1)-size(B,1)#loop for template matching
            for j = 1:size(A,2)-size(B,2)
                corr_map(i,j) = ssi((A(i:i+size(B,1)-1,j:j+size(B,2)-1))(:),B(:));
                corr_map2(i,j) = mse((A(i:i+size(B,1)-1,j:j+size(B,2)-1))(:),B(:));
                waitbar_val = ((i-1)*size(A,2) + j )/(prod(sz_1)-prod(sz_2));
                waitbar(waitbar_val,f,sprintf('%f, %f',i,j))
          endfor
            if getappdata(f,'canceling') == 1
              break
            endif        
        endfor
        close(f);      
        maxpt = max(corr_map(:));
        [x,y]=find(corr_map==maxpt);
        if size(x) ==1
          printf('The pic shows resemblence at (%f,%f)\n',x,y)
          printf('\n')
          maxpt2 = corr_map2(x,y);
          retval = [maxpt2 maxpt];
        endif
        maxpt2 = corr_map2(x,y);      
        if (maxpt > .996 || maxpt2 < .04) && (size(x)==1)
          figure,imagesc(B2);title('Target Image');colormap(gray);axis image#to plot the corresponding area in colour and other area in grayscale
          grayA = rgb2gray(A2);
          Res   = A2;
          Res(:,:,1)=grayA;
          Res(:,:,2)=grayA;
          Res(:,:,3)=grayA;
          Res(x:x+size(B,1)-1,y:y+size(B,2)-1,:)=A2(x:x+size(B,1)-1,y:y+size(B,2)-1,:);
          figure,imagesc(Res);
        else
        disp('The program failed to find the match:')
        printf('\n')
        endif   
       printf('\n')
       printf('ssi for hue:%f \nMSE for hue:%f\n', maxpt, maxpt2)
       printf('\n')
      elseif  size(img2,3) ==1 && size(img1,3) == 1           
        [sz_1, sz_2, A, B] = selector(img1,img2);
        corr_map = zeros([size(A,1),size(A,2)]);
        f = waitbar(0,'1','Name','finding template...',...
            'createcancelbtn','setappdata(gcbf,''canceling'',1)');#setting the progress bar
        for i = 1:size(A,1)-size(B,1)
            for j = 1:size(A,2)-size(B,2)
                corr_map(i,j) = ssi((A(i:i+size(B,1)-1,j:j+size(B,2)-1))(:),B(:));
                corr_map2(i,j) = mse((A(i:i+size(B,1)-1,j:j+size(B,2)-1))(:),B(:));
                waitbar_val = ((i-1)*size(A,2) + j )/(prod(sz_1)-prod(sz_2));
                waitbar(waitbar_val,f,sprintf('%f, %f',i,j))
          endfor
          if getappdata(f,'canceling') == 1
              break
          endif        
        endfor
        close(f);      
        maxpt = max(corr_map(:));
        [x,y]=find(corr_map==maxpt);
        if size(x) ==1
          printf('\n')
          printf('The pic shows resemblence at (%f,%f)\n',x,y)
          printf('\n')
          maxpt2 = corr_map2(x,y);
        endif
      elseif  size(img2,3) ==1 && size(img1,3) == 3
        img1 = rgb2gray(img1);#converting colour image to grey as one of the input image would be grey
        [sz_1, sz_2, A, B] = selector(img1,img2);
        corr_map = zeros([size(A,1),size(A,2)]);
        f = waitbar(0,'1','Name','finding template...',...
            'createcancelbtn','setappdata(gcbf,''canceling'',1)');#setting the progress bar
        for i = 1:size(A,1)-size(B,1)
            for j = 1:size(A,2)-size(B,2)
                corr_map(i,j) = ssi((A(i:i+size(B,1)-1,j:j+size(B,2)-1))(:),B(:));
                corr_map2(i,j) = mse((A(i:i+size(B,1)-1,j:j+size(B,2)-1))(:),B(:));
                waitbar_val = ((i-1)*size(A,2) + j )/(prod(sz_1)-prod(sz_2));
                waitbar(waitbar_val,f,sprintf('%f, %f',i,j));
                if getappdata(f,'canceling') == 1
                  break
                endif
          endfor        
        endfor
        close(f);      
        maxpt = max(corr_map(:));
        [x,y]=find(corr_map==maxpt);
        if size(x) ==1#since there is a chance the template can match at more than 2 location
          printf('\n')
          printf('The pic shows resemblence at (%f,%f)\n',x,y)
          printf('\n')
          maxpt2 = corr_map2(x,y);
          retval = [maxpt2 maxpt];
        endif
      elseif  size(img2,3) ==3 && size(img1,3) == 1
        img2 = rgb2gray(img2);
        [sz_1, sz_2, A, B] = selector(img1,img2);
        corr_map = zeros([size(A,1),size(A,2)]);
        f = waitbar(0,'1','Name','finding template...',...
            'createcancelbtn','setappdata(gcbf,''canceling'',1)');#setting the progress bar
        for i = 1:size(A,1)-size(B,1)
            for j = 1:size(A,2)-size(B,2)
                corr_map(i,j) = ssi((A(i:i+size(B,1)-1,j:j+size(B,2)-1))(:),B(:));
                corr_map2(i,j) = mse((A(i:i+size(B,1)-1,j:j+size(B,2)-1))(:),B(:));
                waitbar_val = ((i-1)*size(A,2) + j )/(prod(sz_1)-prod(sz_2));
                waitbar(waitbar_val,f,sprintf('%f, %f',i,j));
          endfor
          if getappdata(f,'canceling') == 1
              break
            endif        
        endfor
        close(f);     
        maxpt = max(corr_map(:));
        [x,y]=find(corr_map==maxpt);
        if size(x) ==1
          printf('\n')
          printf('The pic shows resemblence at (%f,%f)\n',x,y)
          printf('\n')
          maxpt2 = corr_map2(x,y);
        endif
        
    endif
    
    if size(x) ==1 &&size(y) ==1
      printf('\n')
      printf('The SSI and MSE for the most matching region is\nMSE :%f\nSSI :%f\n', maxpt2,maxpt)
      retval = [maxpt2 maxpt];
      printf('\n')
    else
      retval = [maxpt];
      printf('\n')
      disp('the two pic shows correlation in more than one point. Do you want them all?')
      y_n1 = yes_or_no();
      global y_n;
      y_n = y_n1;
      printf('\n')
      if y_n ==1
        global raw;
        raw = zeros(size(x),4);
        raw(:,1:2) = [x y];
        for i =  1:size(x)
          raw(i,3) = corr_map2(x(i),y(i));
          raw(i,4) = corr_map(x(i),y(i));
        endfor
        printf('\n')
        disp('A n x 4 matrix "raw" has been created. First column contains the x coordinates and second column contains y coordinate. The third and fourth column contains the MSE and SSI of the corresponding position respetively')
        printf('\n')        
      endif
    endif
  endif
  endif
endfunction
function retval = compare(img1,img2)
  #this function is used to filter the condition where this code will break
  if (rows(img1) < rows(img2) && columns(img1) > columns(img2))||(rows(img1) > rows(img2) && columns(img1) < columns(img2))
    disp('Error: dimension mismatch')
    return
  endif
  retval = compare_process(img1,img2);
endfunction
similarity = compare(img1,img2);

if y_n == 1# to print some information on MSE and SSI
  printf('\n Note:\n')
  disp('The value of SSI indicates the similarity of the images, higher ssi means more similar.')
  disp('Higher value(above .9986) of SSI along with 0(approx) MSE indicates identical image.')
  disp('For an image with high SSI and high MSE, the value of SSI is more reliable.')
endif
#asking the user to analyse the same image again as the image may not have been in right orientation when loaded
disp('The given two images have been analysed. Stll if one image is the rotated of another the answer might not be optimum')
disp('Do you want rotate or flip one image and try again?')
y1= yes_or_no();
while y1 == 1#while loop to rerun the analysis on the same image set after rotating or flipping or both
  deg = input('Enter no. of 90 degrees rotation(0 to switch for flipping)');
  if deg == 0;
    dire = input('Enter the direction of flip(1 for flipping with horizontal axis,2 for flipping around vertical axis)');
    img_1 = flip(img1,dire);
    deg = input('The image is flipped. Do you want to continue or rotate the flipped image. Enter 0 to continue or the number of 90 degree rotations to rotate')    
    img_1 = rot90(img_1,deg);
    similarity1 = compare(img_1,img2);
  else
    img_1 = rot90(img1,deg);
    similarity1 = compare(img_1,img2);   
  endif
  disp('Do you want rotate or flip one image and try again?')
  y1 = yes_or_no();  
endwhile
disp('Run the comparsion for two other images?')
again = yes_or_no();#Question to rerun the code as it is easier than rerunning manually
printf('\n')
if again == 1
  printf('\n')
  printf('\n')
  clear-all
  run imagecomp.m
endif