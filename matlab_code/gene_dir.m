function dir = gene_dir(point_num,dim)
% generate directions
% dir = zeros(point_num,dim);
dir = [];
point_num_real = point_num;
even_point_num = floor(point_num_real^(1/(dim-1)));
bias_num = point_num_real - even_point_num^(dim-1);
cnt_dir = 0;
for i = 1:even_point_num
    if i == round(even_point_num / 2)
        point_num2 = even_point_num + bias_num; 
        p = point_num2;
    else
        point_num2 = even_point_num;
        p = point_num2;
    end
    if dim == 2
        dir(i,:) = [cos(2*pi*i/even_point_num) sin(2*pi*i/even_point_num)];
    elseif dim == 3
        while p >= 1
            dir(cnt_dir+1,:) = [sin(pi*i/(even_point_num+1))*cos(2*pi*p/point_num2),...
                sin(pi*i/(even_point_num+1))*sin(2*pi*p/point_num2),...
                cos(pi*i/(even_point_num+1))]; 
            cnt_dir = cnt_dir + 1;
            p = p - 1;
        end
    elseif dim == 4
        for i2 = 1:even_point_num
            if i2 > 1
                p = even_point_num;
            end
            while p >= 1
                angle1 = pi*i/(even_point_num+1);
                angle2 = pi*i2/(even_point_num+1);
                angle3 = 2*pi*p/point_num2;
                dir(cnt_dir+1,:) = [sin(angle1)*sin(angle2)*cos(angle3),...
                    sin(angle1)*sin(angle2)*sin(angle3),...
                    sin(angle1)*cos(angle2),...
                    cos(angle1)]; 
                cnt_dir = cnt_dir + 1;
                p = p - 1;
            end
        end
    elseif dim == 5
        for i2 = 1:even_point_num
            for i3 = 1:even_point_num
                if i2 > 1 || i3 > 1
                    p = even_point_num;
                end
                while p >= 1
                    angle1 = pi*i/(even_point_num+1);
                    angle2 = pi*i2/(even_point_num+1);
                    angle3 = pi*i3/(even_point_num+1);
                    angle4 = 2*pi*p/point_num2;
                    dir(cnt_dir+1,:) = [sin(angle1)*sin(angle2)*sin(angle3)*cos(angle4),...
                        sin(angle1)*sin(angle2)*sin(angle3)*sin(angle4),...
                        sin(angle1)*sin(angle2)*cos(angle3),...
                        sin(angle1)*cos(angle2),...
                        cos(angle1)]; 
                    cnt_dir = cnt_dir + 1;
                    p = p - 1;
                end
            end
        end
    end
end
end
