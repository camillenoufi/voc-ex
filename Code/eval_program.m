%Edit before use:

    audioDir= '/usr/ccrma/media/projects/jordan/Datasets/DAMP-AG/eval_audio_trimmed'; %directory where audio are stored
    trainingDir= '/usr/ccrma/media/projects/jordan/Datasets/DAMP-AG/eval_audio_training'; %directory where training audio are stored
    responseDir= '/usr/ccrma/media/projects/jordan/Datasets/DAMP-AG/eval_audio_responses'; %directory where participant response files are stored
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

     
    %% user intro
    
    clc
    pNum= input ( ' Enter your personal evaluator ID and press ENTER: ', 's'); %Record participant number
    pDelimiter = '-';
    pSaveTag = strcat(pDelimiter,pNum);
    clc
    
    % --------------------- DISPLAY INTRODUCTION -----------------
    % Elena fill in
    disp ( ' In this listening evaulation, you will hear .... ');

    % ---------------------------------------------------------------
    
    %% create/get file list of tracks participant hasn't evaluated
    
    tracksToEvaluate_savename = fullfile(responseDir,strcat(pNum,'_unseenTracks','.mat')); %filepath to list of audio files the coder has yet to see/evaluate
    if exists(tracksToEvaluate_savename,'file')
        tracksToEvaluate_list = load(tracksToEvaluate_savename);
    else   
        tracksToEvaluate_list = dir(fullfile(audioDir,'*.flac'));
        save(tracksToEvaluate_savename,'tracksToEvaluate_list');
    end
    
    %% Training Section
    
    % Elena fill in 
    disp ( 'Press ENTER to begin Training Set' );
    pause; 
    clc
    
    
%% Evaluation Section: load audio tracks, ask questions, save responses, until user quits
        
    cd(audioDir); %Change to directory containing stimuli

    rng('shuffle'); %Shuffle random number seed
    
        
    %Start test:
    
    disp ( ' Press ENTER to begin the evaluation'); 
    disp ( ' ' );
    disp ( ' ' );
    disp ( ' ' );
    pause;
    
    i = 1;
    continueFlag = 1;
    
    while continueFlag
        
        clc 
        message= strcat('Track #:', num2str(i), ':'); %Create on screen message to indicate what number recording participant is currently on
        disp(message); %display message
        
        nfiles = size(tracksToEvaluate_list,1); % get number of files participant can still evaluate
        j = randi(nfiles); % select a random track from current list
        thisTrack = tracksToEvaluate_list(j);
        [path, trackID, ext] = fileparts(thisTrack.name); % get unique file name
        
        % check how many times this track has been evaluated already (needto do this inside the loop in case other coders are working at the same time!)
        % **** there is probably a much simpler/better way to do this!
        evaluated_list = dir(fullfile(responseDir,'*.mat'));
        evaluated_names = {evaluated_list.names}; %convert list of file names to cell
        evaluated_names = cellfun(@(s) strsplit(s, '-'), evaluated_names, 'UniformOutput', false);  %split all filename string into two parts: unique name + participant number
        evaluated_names = evaluated_names{:,1}; %only keep first column
        match_idx = find(strcmp(evaluated_names,trackID)); % find all occurences of unique file name in list of response names
        nEvals = length(match_idx); % count how many occurences

        % if 3 or more evals complete, remove from the coder's list of files to do
        if (nEvals >= 3)
            tracksToEvaluate_list(j) = [];
            save(tracksToEvaluate_savename,'tracksToEvaluate_list');
            
        % else, evaluate the track
        else
            [y,Fs] = audioread(thisTrack.name); % upload audio file
            sound(y, Fs); %Play audio file

            % ------------------- ELENA: List questions options, and response ranges ----------------
            disp(' Evaluation Metrics: ');
            disp('    ');
            disp('1:  ');
            disp('2: ');
            disp('3: ');
            disp('4: ');
            disp('    ');
            pause(15); %length of audio snippet

            % ------------ ELENA: Ask questions ------------------------

            % Question 1:
            disp('Question 1: ...'); %Display question
            disp('Rating Scale: ... ');
            response = input('Type your response and press ENTER', 's'); %take in response

            rMin = 1; rMax = 7; % Define acceptable response range for question
            while ((isempty(response))||(response<rMin)||(response>rMax))   %Catch incorrect response
                response = input ( 'Please enter either 1, 2, 3, or 4 as your response  ');
            end
            % put the response into the structure
            R.age = response;
            
            % Question 2:
            disp('Question 2: ...'); %Display question
            disp('Rating Scale: ... ');
            response= input('Type your response and press ENTER', 's'); %take in response

            rMin = 1; rMax = 7; % Define acceptable response range for question 2
            while ((isempty(response))||(response<rMin)||(response>rMax))   %Catch incorrect response
                response = input ( 'Please enter either 1, 2, 3, or 4 as your response  ');
            end
            % put the response into the structure
            R.gender = response;
            
            % Question 3:
            disp('Question 1: ...'); %Display question
            disp('Rating Scale: ... ');
            response= input('Type your response and press ENTER', 's'); %take in response

            rMin = 1; rMax = 7; % Define acceptable response range for question 3
            while ((isempty(response))||(response<rMin)||(response>rMax))   %Catch incorrect response
                response = input ( 'Please enter either 1, 2, 3, or 4 as your response  ');
            end
            % put the response into the structure
            R.skill = response;
            
            % Question 4:
            disp('Question 1: ...'); %Display question
            disp('Rating Scale: ... ');
            response= input('Type your response and press ENTER', 's'); %take in response

            rMin = 1; rMax = 7; % Define acceptable response range for question 4
            while ((isempty(response))||(response<rMin)||(response>rMax))   %Catch incorrect response
                response = input ( 'Please enter either 1, 2, 3, or 4 as your response  ');
            end
            % put the response into the structure
            R.enjoy = response;
            
            % Question 5:
            disp('Comments: ...');
            response= input('Type any additional pertinent comments (only if necessary) and press ENTER', 's'); %take in response

            % put the response into the structure
            R.enjoy = response;
            
            % save responses to file
            R.datetime = datestr(now); %add time stamp
            savename = strcat(trackID, '-', pNum, '.mat'); %create savename
            savepath = fullfile(responseDir,savename);
            save(savepath,'R');
           
            
            % remove track from coder's to do list
            tracksToEvaluate_list(j) = [];
            save(tracksToEvaluate_savename,'tracksToEvaluate_list');
            
            i = i+1;
            
            % ask if coder wants to continue evaluating
            continueFlag = input('Would you like to evaluate another track? (enter 1 for YES / enter 0 to save and QUIT )', 's'); %take in response
            
        end
    end
    
    