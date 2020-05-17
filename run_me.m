% Predictive decision feedback equalizer
% References: See Section 5.1.7 in the book "Digital Communications and
% Signal Processing" by K Vasudevan
clear all
close all
clc
training_len = 10^4; %length of the training sequence
snr_dB = 10; % snr in dB
ff_filter_len = 30; % feedforward filter length
fb_filter_len = 20; % feedback filter length
data_len = 10^5; % length of the data sequence

% snr parameters
snr = 10^(0.1*snr_dB);
noise_var = 1/(2*snr); % noise variance
% --------------- training phase ------------------------------------------
% source
training_a = randi([0 1],1,training_len);

% bpsk mapper
training_seq = 1-2*training_a;

% impulse response of the channel
fade_chan = [0.9 0.1 0.1 -0.1 ]; 
fade_chan = fade_chan/norm(fade_chan);
chan_len = length(fade_chan);

% awgn
noise = normrnd(0,sqrt(noise_var),1,training_len+chan_len-1);

% channel output
chan_op = conv(fade_chan,training_seq)+noise;

% ------------ LMS update of feedforward filter ---------------------------
ff_filter = zeros(1,ff_filter_len); % feedforward filter initialization

% autocorrelation of received sequence at zero lag
Rvv0 = (chan_op*chan_op')/(training_len+chan_len-1);
% maximum step size
max_step_size = 2/(ff_filter_len*Rvv0);
step_size = 0.125*max_step_size; % step size

for i1=1:training_len-ff_filter_len+1
    ff_filter_ip = fliplr(chan_op(i1:i1+ff_filter_len-1));%equalizer input
    error = ff_filter*ff_filter_ip.' -training_seq(i1+ff_filter_len-1);% instantaneous error
    ff_filter = ff_filter - step_size*error*ff_filter_ip;
end

% mean squared error of interference term
u = conv(ff_filter,chan_op);
u = u(1:training_len);
mse = mean((u-training_seq).^2);

%-----------LMS update of feedbackfilter ----------------------------------
fb_filter = zeros(1,fb_filter_len); % feedback filter initialization
fb_filter_ip = zeros(1,fb_filter_len); % feedback filter input vector

% maximum step size
max_step_size = 2/(fb_filter_len*mse);
step_size = 0.125*max_step_size;

for i1=1:training_len
    quantizer_ip = u(i1)-fb_filter*fb_filter_ip.';
    % hard decision
    quantizer_op = 1-2*(quantizer_ip<0);
    error = quantizer_ip-training_seq(i1);
    
    fb_filter = fb_filter+step_size*fb_filter_ip*error;
    fb_filter_ip = [(u(i1)-quantizer_op) fb_filter_ip(1:end-1)];
end

%------------- data transmission phase-------------------------------------
% source
data_a = randi([0 1],1,data_len);

% bpsk mapper
data_seq = 1-2*data_a;

% awgn
noise = normrnd(0,sqrt(noise_var),1,data_len+chan_len-1);

% channel output
chan_op = conv(fade_chan,data_seq)+noise;

% equalization
ff_filter_ip = zeros(1,ff_filter_len); % feedforward input vector
fb_filter_ip = zeros(1,fb_filter_len); % feedback input vector
fb_filter_op = 0; % feedback filter output symbol
dec_data_seq = zeros(1,data_len);

for i1=1:data_len
ff_filter_ip = [chan_op(i1) ff_filter_ip(1:end-1)];
ff_filter_op = ff_filter*ff_filter_ip.';
quantizer_ip = ff_filter_op - fb_filter_op;
% hard decision
dec_data_seq(i1) = 1-2*(quantizer_ip<0);
fb_filter_ip = [ff_filter_op-dec_data_seq(i1) fb_filter_ip(1:end-1)];
fb_filter_op = fb_filter*fb_filter_ip.';        
end

% demapping symbols to bits
dec_a = dec_data_seq<0;

% bit error rate
BER = nnz(dec_a-data_a)/data_len