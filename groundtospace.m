function guk = groundtospace(fk, vertheta)
    % Create an ITU-R P.681-11 channel or Lutz LMS channel
    chan = p681LMSChannel;
    % Set channel properties
    if isa(chan,"p681LMSChannel")
        % For ITU-R P.681 LMS channel
    
        % Environment type
        chan.Environment = "Urban";
        % Carrier frequency (in Hz)
        chan.CarrierFrequency = fk;
        % Elevation angle with respect to ground plane (in degrees)
        chan.ElevationAngle = vertheta;
        % Speed of movement of ground terminal (in m/s)
        chan.MobileSpeed = 2;
        % Direction of movement of ground terminal (in degrees)
        chan.AzimuthOrientation = 0;
    else
        % For Lutz LMS channel
    
        % Rician K-factor (in dB)
        chan.KFactor = 5.5;
        % Lognormal fading parameters (in dB)
        chan.LogNormalFading = [-13.6 3.8];
        % State duration distribution
        chan.StateDurationDistribution = "Exponential";
        % Mean state duration (in seconds)
        chan.MeanStateDuration = [21 24.5];
        % Maximum Doppler shift (in Hz)
        chan.MaximumDopplerShift = 2.8538;
    end
    % Sampling rate (in Hz)
    chan.SampleRate = 4000;
    chan.InitialState = "Good";
    chan.FadingTechnique = "Filtered Gaussian noise"; 
    seed = 73;
    chan.RandomStream = "mt19937ar with seed";
    chan.Seed = seed;
    % Set random number generator with seed
    rng(seed);
    % Channel duration (in seconds)
    chanDur = 1/4001;
    % Random input waveform
    numSamples = floor(chan.SampleRate*chanDur)+1;
    in = complex(randn(numSamples,1),randn(numSamples,1));
    % Pass the input signal through channel
    [~,guk,~,~] = step(chan,in);
end


