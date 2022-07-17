%Author : Daniela Cuza

function data = contrast_blur_h_new(data, func, direction, harsh)
    siz=224;
    for pattern = 1:size(data,1)
        data{pattern,1} =imresize(data{pattern,1},[siz siz]);
        data{pattern,2} =imresize(data{pattern,2},[siz siz]);
        trainingImageFinal= uint8(zeros(224,224,3));
        %% contrast 
        
        xval = linspace(0,1,256); % every possible pixel intensity divided by 255
        
        if(func==1) % function 1
            
            % set limits for rand function
            if(direction)
                if(harsh)
                    a = -5;
                    b = -3;
                else
                    a = -2;
                    b = -1;
                end
            else
                if(harsh)
                    a = 2.8;
                    b = 3.8;
                else
                    a = 1.5;
                    b = 2.5;
                end
            end
            
            % k controls the slope of the curve, when k>0 the contrast decreases, when k<0 the contrast increases
            k = a + (b - a)*rand;
            
            yval = ((xval-1/2).*sqrt(1-k/4))./sqrt(1-((xval-1/2).^2).*k)+0.5;
            
        else % function 2
            
            % set limits for rand function
            if(direction)
                if(harsh)
                    a = 1.8;
                    b = 2.3;
                else
                    a = 1.2;
                    b = 1.7;
                end
            else
                if(harsh)
                    a = 0.25;
                    b = 0.5;
                else
                    a = 0.6;
                    b = 0.9;
                end
            end
            
            % q controls the slope of the curve, when q<1 the contrast decreases, when q>1 the contrast increases
            q = a + (b - a)*rand;

            syms y(x)
            y(x) = piecewise(x<1/2, 0.5*(x/0.5)^q, x>1/2, 1-0.5*((1-x)/0.5)^q);

            yval = double(y(xval));
        end
        
        yval = yval*255;
        
        % applying contrast 
        oldIM = data{pattern,1}; %image
        lb = logical(data{pattern,2}); %label
        
        ch1 = uint8(oldIM(:,:,1));
        ch2 = uint8(oldIM(:,:,2));
        ch3 = uint8(oldIM(:,:,3));
        
        ch1 = uint8(yval(ch1+1));
        ch2 = uint8(yval(ch2+1));
        ch3 = uint8(yval(ch3+1));
        lb=yval(lb+1);
        
        %% motion blur
        h = fspecial('motion' , 3+rand*4 , rand*360); % movement filter, U(3,7) pixel in direction U(0,360)
        ch1 = imfilter(ch1,h,'replicate');
        ch2 = imfilter(ch2,h,'replicate');
        ch3 = imfilter(ch3,h,'replicate');
        lb = imfilter(lb,h,'replicate');
        
        %% append modified image at the end of the dataset
        trainingImageFinal(:,:,1) = ch1;
        trainingImageFinal(:,:,2) = ch2;
        trainingImageFinal(:,:,3) = ch3;
        data{pattern,1} = trainingImageFinal;
        data{pattern,2}=lb;
    end
end