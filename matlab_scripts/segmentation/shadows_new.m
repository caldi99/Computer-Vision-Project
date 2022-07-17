%Author : Daniela Cuza
function data = shadows_new(data)
    siz=224;
    for pattern = 1:size(data,1)               
        %% shadows 
        data{pattern,1} =imresize(data{pattern,1},[siz siz]);
        data{pattern,2} =imresize(data{pattern,2},[siz siz]);
       
        
        xval = linspace(0,1,224); % every column number normalized between 0 and 1
        
        direction = randi(2)-1; % 1: l->r, 0: r->l
        
        % function darkens half of the image
        if(direction)
           yval = 0.2+((xval./0.5).^(1/2)).*0.8;
        else
           yval = 0.2+(((-xval+1)./0.5).^(1/2)).*0.8;
        end
        
        % cap values of the function at 1, so that after the midpoint of
        % the image is reached, pixel intensities get multiplied by
        yval(yval>1)=1; 
        
        % applying shadows 
        oldIM = data{pattern,1}; %image
        ch1IM = oldIM(:,:,1);
        ch2IM = oldIM(:,:,2);
        ch3IM = oldIM(:,:,3);
        lb = logical(data{pattern,2}); %label
        
        i=1;
        for i = 1 : 224
            ch1IM(:,i) = ch1IM(:,i)*yval(i);
            ch2IM(:,i) = ch2IM(:,i)*yval(i);
            ch3IM(:,i) = ch3IM(:,i)*yval(i);
            lb(:,i)=lb(:,i)*yval(i);
        end
        
        newIM(:,:,1) = ch1IM;
        newIM(:,:,2) = ch2IM;
        newIM(:,:,3) = ch3IM;
        %% append modified image at the end of the dataset
        data{pattern,1} = newIM;
        data{pattern,2}=lb;
    end
end