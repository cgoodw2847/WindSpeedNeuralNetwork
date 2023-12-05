% Given data
col1 = [zeros(24, 1); 1; zeros(25, 1)];
col2 = ones(50, 4);

% Combine columns into neural_net array
neural_net = [col1, col2, col1];

% Updated parameters
circle_diameter = 20;
vertical_spacing = 2;
column_spacing = 120;
line_thickness = 0.15;

% Create figure
figure;

% Loop through each column (except the last one)
for col = 1:size(neural_net, 2) - 1
    % Loop through each row in the current column
    for row = 1:size(neural_net, 1)
        % Check if there is a circle (1) at this position
        if neural_net(row, col) == 1
            % Connect with black line to all circles in the next column
            next_col = col + 1;
            for next_row = 1:size(neural_net, 1)
                if neural_net(next_row, next_col) == 1
                    % Calculate line endpoints
                    x_center = (col - 1) * (circle_diameter + column_spacing) + circle_diameter / 2;
                    y_center = (row - 1) * (circle_diameter + vertical_spacing) + circle_diameter / 2;
                    next_x_center = (next_col - 1) * (circle_diameter + column_spacing) + circle_diameter / 2;
                    next_y_center = (next_row - 1) * (circle_diameter + vertical_spacing) + circle_diameter / 2;

                    % Draw black line with adjusted line width
                    line([x_center, next_x_center], [y_center, next_y_center], 'Color', 'k', 'LineWidth', line_thickness);
                end
            end
        end
    end
end

% Loop through each column (including the last one) to draw circles
for col = 1:size(neural_net, 2)
    % Loop through each row in the current column
    for row = 1:size(neural_net, 1)
        % Check if there is a circle (1) at this position
        if neural_net(row, col) == 1
            % Calculate circle center coordinates
            x_center = (col - 1) * (circle_diameter + column_spacing) + circle_diameter / 2;
            y_center = (row - 1) * (circle_diameter + vertical_spacing) + circle_diameter / 2;

            % Draw blue circle
            rectangle('Position', [x_center - circle_diameter/2, y_center - circle_diameter/2, circle_diameter, circle_diameter], ...
                'Curvature', [1, 1], 'FaceColor', 'b');
        end
    end
end

% Set axis equal and tight
axis equal;
axis tight;

axis off;
