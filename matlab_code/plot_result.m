function plot_result(vertex,str1,str2,str3,wline,alpha,line_flag)
dim = size(vertex,2);
if dim == 2
    ch_vert = convhulln(vertex);
    for i = 1:size(ch_vert,1)
        line(vertex(ch_vert(i,:),1),vertex(ch_vert(i,:),2),...
            'Color',str1,...
            'LineWidth',wline,...
            'LineStyle',str2);
        hold on;
    end
elseif dim == 3
    ch_vert = convhulln(vertex);
    x = vertex(:,1);
    y = vertex(:,2);
    z = vertex(:,3);
    warning off
    tr = TriRep(ch_vert,x,y,z);
    warning on;
    fe = featureEdges(tr,pi/70)';
    trisurf(tr, 'FaceColor', str3, 'EdgeColor',str1, 'FaceAlpha', alpha); 
    hold on; 
    if line_flag
        plot3(x(fe), y(fe), z(fe), str2, 'LineWidth',wline); 
        hold on;
    end
%     trisurf(ch_vert,vertex(:,1),vertex(:,2),vertex(:,3),...
%         'FaceColor',str3,'EdgeColor',str1,'FaceAlpha',alpha);
end