function [circleParaXYR] = cluster_points(data, criteria)  
    number = size(data);
    criteria = 2 * criteria^2;
    index = ones(number(1), 1);
    for i = 1:number-1
        if index(i) == 1
            for j = i+1:number
                distance = (data(i, 1) - data(j, 1))^2 +(data(i, 2) - data(j, 2))^2;
                if distance < criteria
                    index(j) = 0;
                end
            end
        end
    end
    circleParaXYR = data(index>0, :);

end
