%% Experimenting with the price data in order to find a good prediction model 
clc 
clear 

% Load Data and write in timetable
Data = readmatrix('191228_prices_ProKoMo.xlsx','Range','B43826:C78889');
start_time = datetime('01.01.15','Inputformat','dd.MM.yy','TimeZone','UTC'); 
time_steps = hours(1);
TT_raw = timetable(Data(:,1),Data(:,2),(Data(:,1) - Data(:,2)),'TimeStep',time_steps,'StartTime',start_time, 'VariableNames',{'PriceActual','PriceForecast','ForecastingError'});

% Data preparation
% Find NaN entries
[rownbr_missing_price_forecast, ] = find(isnan(TT_raw(:,:).PriceForecast));
[rownbr_missing_price_actual, ] = find(isnan(TT_raw(:,:).PriceActual));
date_missing_price_forecast = TT_raw.Time(rownbr_missing_price_forecast);
date_missing_price_actual = TT_raw.Time(rownbr_missing_price_actual);
% delete NaNs 
% unique code for every day (year;month;day)
Time_matrix = datevec(TT_raw.Time);
code_day = Time_matrix(:,1) * 10000  + Time_matrix(:,2) * 100 + Time_matrix(:,3);
days_with_NaN = unique([code_day(rownbr_missing_price_forecast,:); code_day(rownbr_missing_price_actual,:)]);
ind_locial = find(ismember(code_day, days_with_NaN));
TT_raw(ind_locial,:).ForecastingError = NaN(size(TT_raw(ind_locial,:).ForecastingError));
TT = TT_raw(~isnan(TT_raw.ForecastingError),:);
[weekday_code,weekday] = weekday(TT.Time);
Time_matrix = datevec(TT.Time);
[years,ia,idx_years] = unique(year(TT.Time));
%% Looking at available data  

% MSE and MAE of Forecasting Error
mse_FE = mean((TT.ForecastingError.^2));
mae_FE = mean(abs(TT.ForecastingError));

figure_nbr = 1; 
% Plot Price Forecast BTU and Actual Price 
figure(figure_nbr)
tiledlayout(size(years,1),1)
for i = 1:size(years,1)
    nexttile
    plot(TT(year(TT.Time) == years(i,:),:).Time,TT(year(TT.Time) == years(i,:),:).PriceForecast)
    hold on 
    plot(TT(year(TT.Time) == years(i,:),:).Time,TT(year(TT.Time) == years(i,:),:).PriceActual)
    xlabel('Time')
    ylabel('Price')
    set(gca,'FontSize',14)
    legend('Price Forecast','Price Actual','Location','northeastoutside')
end
figure_nbr = figure_nbr + 1;

% Plot Error of Price Forecast 
figure(figure_nbr)
tiledlayout(size(years,1),1)
for i = 1:size(years,1)
    nexttile
    plot(TT(year(TT.Time) == years(i,:),:).Time,TT(year(TT.Time) == years(i,:),:).ForecastingError)
    xlabel('Time')
    ylabel('Price')
    set(gca,'FontSize',14)
    legend('Error','Location','northeastoutside')
end
figure_nbr = figure_nbr + 1;

% Load CSV with german holidays
Holiday_table = readtable('HolidaysGermany15to19.csv','PreserveVariableNames',true);
Holidays = datetime(table2array(Holiday_table(:,1)),'Inputformat','dd.MM.yy', 'TimeZone','UTC');

% Inspect effects from holidays 
holidays_all = ismember(TT.Time, Holidays);
for i = 1:(size(holidays_all,1)-1)
    if (holidays_all(i,1) == 1 && day(TT(i+1,:).Time) == day(TT(i,:).Time)) 
       holidays_all(i+1,:) = 1;
    end
end

holidays_vec_values = holidays_all.*TT.ForecastingError;
holidays_vec_values(holidays_vec_values == 0) = NaN;
% plot error and holidays in one plot 
figure(figure_nbr)
tiledlayout(size(years,1),1)
for i = 1:size(years,1)
    nexttile
    plot(TT(year(TT.Time) == years(i,:),:).Time,TT(year(TT.Time) == years(i,:),:).ForecastingError)
    hold on 
    plot(TT(year(TT.Time) == years(i,:),:).Time,holidays_vec_values(year(TT.Time) == years(i,:),:))
    xlabel('Time')
    ylabel('Price')
    set(gca,'FontSize',14)
    legend('Error','Holidays','Location','northeastoutside')
end
figure_nbr = figure_nbr + 1;

%% Test stationarity 

% Augmented Dickey Fuller Tests
[h_adf_raw,pValue_adf_raw,stat_adf_raw,cValue_adf_raw,reg_adf_raw] = adftest(TT.ForecastingError,'alpha',0.05);
[h_t_adf_raw,pValue_t_adf_raw,stat_t_adf_raw,cValue_t_adf_raw,reg_t_adf_raw] = adftest(TT.ForecastingError,'model','ts','alpha',0.05);
% Result: h = 1: reject H0, so reject the possibility of a unit root 
% KPSS Test
sqrt = sqrt(size(TT.ForecastingError,1));
[h_kpss_raw,pValue_kpss_raw] = kpsstest(TT.ForecastingError, 'lags',round(sqrt),'alpha',0.05);
% Result: h = 1: reject H0 (H0 is trend stationarity) 

%% Classical decomposition 

% TREND
% Mean value in every year of prices and error 
mean_years_Error = accumarray(idx_years, TT.ForecastingError,[],@mean);
mean_years_Price_Act = accumarray(idx_years, TT.PriceActual,[],@mean);
mean_years_Price_Forecast = accumarray(idx_years, TT.PriceForecast,[],@mean);
% Result: 

% SEASONALITY 

% Monthly - so calculate mean of every month in every year and plot it 
[coding_every_month,mean_every_month,coding_month,mean_month]  = myfun_mean_every_month(TT);
myfun_plot_mean_yearly(mean_every_month, mean_month, coding_month, figure_nbr)
figure_nbr = figure_nbr + 1; 

% Weekly - - so calculate mean of every weekday in every month and every year and plot it
[coding_weekday,ia ,idx_weekday] = unique(weekday_code);
mean_weekdays = accumarray(idx_weekday, TT.ForecastingError,[],@mean);
[mean_weekday, coding_weekdays] = myfun_mean_of_weekday(TT,weekday_code);
myfun_plot_mean_weekday(mean_weekday, coding_weekdays, figure_nbr)
figure_nbr = figure_nbr + 1; 

% Daily - so calculate mean of every hour for the different weekdays and plot it  
hour = hour(TT.Time);
[coding_hour,ia ,idx_hour] = unique(hour);
mean_hour = accumarray(idx_hour, TT.ForecastingError,[],@mean);
[mean_hourly_weekday,coding_hours, coding_hour_vec] = myfun_mean_hourly_per_weekday(TT);
myfun_plot_hourly_mean(mean_hourly_weekday,figure_nbr)
figure_nbr = figure_nbr + 1; 

% Based on results adjust data by subtracting the seasonality 
%% Plots of ACF and PACF for different Data and Models 
% e.g. deseasonalized data, SARIMA differenced data, 24h Model 

% ACF and PACF of the ForecastingError 
figure(figure_nbr)
subplot(2,1,1)
autocorr(TT.ForecastingError,'NumLags',192,'NumSTD',2)
title('ACF of the Forecasting Error')
xlabel('Number of Lags')
ylabel('ACF')
set(gca,'FontSize',13)
subplot(2,1,2)
parcorr(TT.ForecastingError,'NumLags',192,'NumSTD',2)
title('PACF of the Forecasting Error')
xlabel('Number of Lags')
ylabel('PACF')
set(gca,'FontSize',13)
figure_nbr = figure_nbr + 1; 

% ACF and PACF of the deseasonalized ForecastingError 
% !!! Hier Daten die geplotted werden ändern 
% figure(figure_nbr)
% subplot(2,1,1)
% autocorr(TT.ForecastingError,'NumLags',192,'NumSTD',2)
% title('ACF of the Forecasting Error')
% xlabel('Number of Lags')
% ylabel('ACF')
% set(gca,'FontSize',13)
% subplot(2,1,2)
% parcorr(TT.ForecastingError,'NumLags',192,'NumSTD',2)
% title('PACF of the Forecasting Error')
% xlabel('Number of Lags')
% ylabel('PACF')
% set(gca,'FontSize',13)
% figure_nbr = figure_nbr + 1; 
 
% SARIMA Model 

% Seasonal differencing 
diff_24 = LagOp({1 -1},'Lags',[0,24]); 
ForecastingError_diff_24 = filter(diff_24,TT.ForecastingError);
diff_168 = LagOp({1 -1},'Lags',[0,168]); 
ForecastingError_diff_168 = filter(diff_168,TT.ForecastingError);

% ACF and PACF of the seasonally differenced ForecastingError (seasonality
% 24 for daily recurring effects, seasonality 168 for weekly recurring
% effects) 
figure(figure_nbr)
subplot(2,2,1)
autocorr(ForecastingError_diff_24,'NumLags',192,'NumSTD',2)
title('ACF of the Forecasting Error (D=1,s=24)')
xlabel('Number of Lags')
ylabel('ACF')
set(gca,'FontSize',13)
subplot(2,2,2)
parcorr(ForecastingError_diff_24,'NumLags',192,'NumSTD',2)
title('PACF of the Forecasting Error (D=1,s=24)')
xlabel('Number of Lags')
ylabel('PACF')
set(gca,'FontSize',13)
subplot(2,2,3)
autocorr(ForecastingError_diff_168,'NumLags',192,'NumSTD',2)
title('ACF of the Forecasting Error (D=1,s=168)')
xlabel('Number of Lags')
ylabel('ACF')
set(gca,'FontSize',13)
subplot(2,2,4)
parcorr(ForecastingError_diff_168,'NumLags',192,'NumSTD',2)
title('PACF of the Forecasting Error (D=1,s=168)')
xlabel('Number of Lags')
ylabel('PACF')
set(gca,'FontSize',13)
figure_nbr = figure_nbr + 1; 

% Short Test how autocorrelation will look like for 24h Model 

FE = (TT.ForecastingError)'; 
FE = (reshape(FE,24,(size(FE,2)/24)))';

% ACF and PACF of the ForecastingError exemplary for hour 0 and 12

figure(figure_nbr)
subplot(2,2,1)
autocorr(FE(:,1),'NumLags',20,'NumSTD',2)
title('ACF of the Forecasting Error hour 0')
xlabel('Number of Lags')
ylabel('ACF')
set(gca,'FontSize',13)
subplot(2,2,2)
parcorr(FE(:,1),'NumLags',20,'NumSTD',2)
title('PACF of the Forecasting Error hour 0')
xlabel('Number of Lags')
ylabel('PACF')
set(gca,'FontSize',13)
subplot(2,2,3)
autocorr(FE(:,13),'NumLags',20,'NumSTD',2)
title('ACF of the Forecasting Error hour 12')
xlabel('Number of Lags')
ylabel('ACF')
set(gca,'FontSize',13)
subplot(2,2,4)
parcorr(FE(:,13),'NumLags',20,'NumSTD',2)
title('PACF of the Forecasting Error hour 12')
xlabel('Number of Lags')
ylabel('PACF')
set(gca,'FontSize',13)
figure_nbr = figure_nbr + 1; 
 

% choose p and q 

%% prediction model 

% compare results from univariate and multivariate Model 

%% Univariate 

end_date_training_data = datetime('31.12.15 23:00:00','Inputformat','dd.MM.yy HH:mm:ss','TimeZone','UTC');
start_date_forecast = datetime('01.01.16 00:00:00','Inputformat','dd.MM.yy HH:mm:ss','TimeZone','UTC');

[a ,forecast_start_date_rownbr] = max(TT.Time == start_date_forecast);
timevec_all_forecasts = TT(forecast_start_date_rownbr:end, :).Time;
forecast_summary = timetable(timevec_all_forecasts, zeros(size(timevec_all_forecasts)), zeros(size(timevec_all_forecasts)), zeros(size(timevec_all_forecasts)) , zeros(size(timevec_all_forecasts)), zeros(size(timevec_all_forecasts)), zeros(size(timevec_all_forecasts)), zeros(size(timevec_all_forecasts)), 'VariableNames',{'Forecast_Deseasonal','MSE_Forecast_Deseasonal', 'Seasonality', 'ErrorForecast', 'IndexTrainDataStart', 'IndexTrainDataEnd','NewForecastError'} );

param_estimates_summary = zeros((size(timevec_all_forecasts,1))/24 ,4);

% BEGIN LOOP
j = days(0);
Mdl_Time_Series = arima(1,0,1);
for i = 1:(size(forecast_summary,1)/24)
    % select training data for loop iteration
    end_date_training_data = end_date_training_data + j;
    start_date_training_data = end_date_training_data - calendarDuration(1,0,0) + hours(1); 
    [a ,forecast_start_date_rownbr] = max(TT.Time == start_date_training_data);
    [a ,forecast_end_date_rownbr] = max(TT.Time == end_date_training_data);
    training_Data_loop = TT(forecast_start_date_rownbr:forecast_end_date_rownbr,:).ForecastingError;

    % what I want to forecast
    start_date_forecast = start_date_forecast + j
    forecast_vec = timetable(zeros(24,1),'TimeStep',hours(1),'StartTime',start_date_forecast, 'VariableNames',{'Seasonality'});
   
        % calculate seasonality of training data
        % seasonality depends on ...
        %[Data_seas_adj, forecast_vec_day_ahead] = myfun_adj_seas(training_Data_loop, forecast_vec_day_ahead);

        % Use deseasonalized data and ARMA(,,) Model for forecast
        [EstMdl_Time_Series_loop, EstParamCov_loop, logL_loop, info_loop] = estimate(Mdl_Time_Series,training_Data_loop);
        [Y, YMSE] = forecast(EstMdl_Time_Series_loop, 24 , training_Data_loop);
        param_estimates_summary(i ,:) = (info_loop.X).';
        forecast_summary(((i-1)*24) + 1 : i*24,:).Forecast_Deseasonal = Y;
        forecast_summary(((i-1)*24) + 1 : i*24,:).MSE_Forecast_Deseasonal = YMSE;
        forecast_summary(((i-1)*24) + 1 : i*24,:).Seasonality = forecast_vec.Seasonality;
        forecast_summary(((i-1)*24) + 1 : i*24,:).IndexTrainDataStart = forecast_start_date_rownbr + zeros(24,1);
        forecast_summary(((i-1)*24) + 1 : i*24,:).IndexTrainDataEnd = forecast_end_date_rownbr + zeros(24,1);
        
    j = caldays(1);

end
forecast_summary.ErrorForecast = forecast_summary.Forecast_Deseasonal + forecast_summary.Seasonality;

%% Multivariate Classical decomposition 

FE = (TT.ForecastingError)'; 
FE = (reshape(FE,24,(size(FE,2)/24)))'; 
end_date_training_data_mv = datetime('31.12.15 00:00:00','Inputformat','dd.MM.yy HH:mm:ss','TimeZone','UTC');
start_date_forecast_mv = datetime('01.01.16 00:00:00','Inputformat','dd.MM.yy HH:mm:ss','TimeZone','UTC');
% create new timetable which captures every hour separately in one row to
% simplify loop  
start_time_mv = datetime('01.01.15 00:00:00','Inputformat','dd.MM.yy HH:mm:ss','TimeZone','UTC'); 
time_steps_mv = days(1);
TT_mv = timetable(FE(:,1),FE(:,2),FE(:,3),FE(:,4),FE(:,5),FE(:,6),FE(:,7),FE(:,8),FE(:,9),FE(:,10),FE(:,11),FE(:,12),FE(:,13),FE(:,14),FE(:,15),FE(:,16),FE(:,17),FE(:,18),FE(:,19),FE(:,20),FE(:,21),FE(:,22),FE(:,23),FE(:,24),'TimeStep',time_steps_mv,'StartTime',start_time_mv, 'VariableNames',{'null','one','two','three','four','five','six','seven','eigth','nine','ten','eleven','twelve','thirteen','fourteen','fifteen','sixteen','seventeen','eighteen','nineteen','twenty','twentyone','twentytwo','twentythree'});

[a ,forecast_start_date_rownbr_mv] = max(TT_mv.Time == start_date_forecast_mv);
timevec_all_forecasts_mv = TT_mv(forecast_start_date_rownbr_mv:end, :).Time;
forecast_summary_mv = NaN(size(timevec_all_forecasts_mv,1),size(FE,2),6); % + 1 because we make on last day forecast until 11 am 01.01.19
param_estimates_summary_mv = NaN(size(timevec_all_forecasts_mv,1),4,24);

% % % BEGIN LOOP
j = days(0);
% just representative arma(1,1); has to be adjusted 
Mdl_Time_Series = arima(1,0,1);
for i = 1:size(forecast_summary_mv,1)
    
    % select training data for loop iteration
    end_date_training_data_mv = end_date_training_data_mv + j;
    start_date_training_data_mv = end_date_training_data_mv - calendarDuration(0,11,30); 
    [a ,forecast_start_date_rownbr] = max(TT_mv.Time == start_date_training_data_mv);
    [a ,forecast_end_date_rownbr] = max(TT_mv.Time == end_date_training_data_mv);
    % what I want to forecast
    start_date_forecast_mv = start_date_forecast_mv + j
    
    % have to go through all different hours separately
    for ii = 1:24
        training_Data_loop_mv = FE(forecast_start_date_rownbr:forecast_end_date_rownbr ,ii);
        forecast_start_date_rownbr 
        forecast_end_date_rownbr 
        ii 
        % ************ ??? **************
         % calculate seasonality of training data
         %[Data_seas_adj, forecast_vec_day_ahead] = myfun_adj_seas(training_Data_loop, forecast_vec_day_ahead);
        
        % estimate model in every loop based on training data 
        [EstMdl_Time_Series_loop, EstParamCov_loop, logL_loop, info_loop] = estimate(Mdl_Time_Series,training_Data_loop_mv);
        % calculate forecast based on estimated model and training data 
        [Y, YMSE] = forecast(EstMdl_Time_Series_loop, 1, training_Data_loop_mv);
        % safe parameter estimates 
        param_estimates_summary_mv(i,:,ii) = (info_loop.X).';
        % safe all relevant information in 3D forecast matrix 
        forecast_summary_mv(i,ii,1) = Y;
        % safe corresponding deseasoanlized data and seasonality 
        %forecast_day_ahead_summary_mv(i,ii,2) = ;
        %forecast_day_ahead_summary_mv(i,ii,3) = ;
        forecast_summary_mv(i,ii,4) = YMSE;
        forecast_summary_mv(i,ii,5) = forecast_start_date_rownbr;
        forecast_summary_mv(i,ii,6) = forecast_end_date_rownbr;
             
    end
    j = caldays(1);
end

%% Multivariate with SARIMA

% copied from above because I changed it in loop above 
end_date_training_data_mv = datetime('31.12.15 00:00:00','Inputformat','dd.MM.yy HH:mm:ss','TimeZone','UTC');
start_date_forecast_mv = datetime('01.01.16 00:00:00','Inputformat','dd.MM.yy HH:mm:ss','TimeZone','UTC'); 
start_time_mv = datetime('01.01.15 00:00:00','Inputformat','dd.MM.yy HH:mm:ss','TimeZone','UTC'); 

[a ,forecast_start_date_rownbr_mv] = max(TT_mv.Time == start_date_forecast_mv);
timevec_all_forecasts_mv_sarima = TT_mv(forecast_start_date_rownbr_mv:end, :).Time;
forecast_summary_mv_sarima = NaN(size(timevec_all_forecasts_mv,1),size(FE,2),4); 
param_estimates_summary_mv_sarima = NaN(size(timevec_all_forecasts_mv,1),6,24);

% BEGIN LOOP
j = days(0);
% just representative arma(1,1); has to be adjusted 
Mdl_Time_Series = arima('ARLags',1,'MALags',1, 'Seasonality', 7, 'SARLags', 7, 'SMALags', 7);
for i = 1:size(forecast_summary_mv_sarima,1)
    
    % select training data for loop iteration
    end_date_training_data_mv = end_date_training_data_mv + j;
    start_date_training_data_mv = end_date_training_data_mv - calendarDuration(0,11,30); 
    [a ,forecast_start_date_rownbr] = max(TT_mv.Time == start_date_training_data_mv);
    [a ,forecast_end_date_rownbr] = max(TT_mv.Time == end_date_training_data_mv);
    % what I want to forecast
    start_date_forecast_mv = start_date_forecast_mv + j
    
    % have to go through all different hours separately
    for ii = 1:24
        
        training_Data_loop_mv = FE(forecast_start_date_rownbr:forecast_end_date_rownbr,ii);
        forecast_start_date_rownbr 
        forecast_end_date_rownbr 
        ii         
        % estimate model in every loop based on training data 
        [EstMdl_Time_Series_loop, EstParamCov_loop, logL_loop, info_loop] = estimate(Mdl_Time_Series,training_Data_loop_mv);
        % calculate forecast based on estimated model and training data 
        [Y, YMSE] = forecast(EstMdl_Time_Series_loop, 1, training_Data_loop_mv);
        % safe parameter estimates 
        param_estimates_summary_mv_sarima(i,:,ii) = (info_loop.X).';
        % safe all relevant information in 3D forecast matrix 
        forecast_summary_mv_sarima(i,ii,1) = Y;
        forecast_summary_mv_sarima(i,ii,2) = YMSE;
        forecast_summary_mv_sarima(i,ii,3) = forecast_start_date_rownbr;
        forecast_summary_mv_sarima(i,ii,4) = forecast_end_date_rownbr;
             
    end
    j = caldays(1);
end


%% VAR Model 

end_date_training_data_mv = datetime('31.12.15 00:00:00','Inputformat','dd.MM.yy HH:mm:ss','TimeZone','UTC');
start_date_forecast_mv = datetime('01.01.16 00:00:00','Inputformat','dd.MM.yy HH:mm:ss','TimeZone','UTC'); 
start_time_mv = datetime('01.01.15 00:00:00','Inputformat','dd.MM.yy HH:mm:ss','TimeZone','UTC'); 

[a ,forecast_start_date_rownbr_mv] = max(TT_mv.Time == start_date_forecast_mv);
timevec_all_forecasts_mv_arma = TT_mv(forecast_start_date_rownbr_mv:end, :).Time;
forecast_summary_varm = NaN(size(timevec_all_forecasts_mv_arma,1),size(FE,2),3); % + 1 because we make on last day forecast until 11 am 01.01.19
forecast_mse_varm = {size(forecast_summary_varm,1),1};
param_estimates_summary_varm = {size(forecast_summary_varm,1),1};

% BEGIN LOOP
j = days(0);
% varm model, 24 time series and p = 1
Mdl_varm = varm(24,1);
for i = 1:size(forecast_summary_mv_sarima,1)
    
    % select training data for loop iteration
    end_date_training_data_mv = end_date_training_data_mv + j;
    start_date_training_data_mv = end_date_training_data_mv - calendarDuration(0,11,30); 
    [a ,forecast_start_date_rownbr] = max(TT_mv.Time == start_date_training_data_mv);
    [a ,forecast_end_date_rownbr] = max(TT_mv.Time == end_date_training_data_mv);
    % what I want to forecast
    start_date_forecast_mv = start_date_forecast_mv + j      
    training_Data_loop_mv = FE(forecast_start_date_rownbr:forecast_end_date_rownbr,:);        
    % estimate model in every loop based on training data 
    EstMdl_Time_Series_loop = estimate(Mdl_varm,training_Data_loop_mv);
    % calculate forecast based on estimated model and training data 
    [Y, YMSE] = forecast(EstMdl_Time_Series_loop, 1, training_Data_loop_mv);
    % safe parameter estimates 
%%%    param_estimates_summary_mv_sarima(i,:) = EstMdl_Time_Series_loop.AR;
    % safe all relevant information in 3D forecast matrix 
    forecast_summary_varm(i,:,1) = Y;
    forecast_summary_varm(i,:,2) = forecast_start_date_rownbr;
    forecast_summary_varm(i,:,3) = forecast_end_date_rownbr;
    
    forecast_mse_varm(i,:) = YMSE; % is same as covariance in estimated model 
             
    j = caldays(1);
end

%% GARCH Model ? 

%% Relative and absolute evaluation of the forecasts 
% Question: which scoring function should we use? I would use MSE or MAE
% since they are symmetric and depend on forecast error only; 
% but whats also often used in energy price forecasts is absolute
% percentage/relative error but since they should only be used for strictly
% positive quantities (here not case) would recommend them (and anyway, we
% don't compare different datasets so this relative evaluation doesn't
% matter anyway I guess
% **** Comparison of the different MSEs against each other **** 

mse_univariate_old = mean((TT(8761:end,:).ForecastingError.^2));
mse_univariate_cdm = mean(((TT(8761:end,:).ForecastingError-forecast_summary(:,:).ErrorForecast).^2));
mse_multivariate_mv = mean(FE(366:end,:).^2);
mse_multivaraite_cdm = mean(reshape((FE(366:end,:)-forecast_summary_mv(:,:,1)),1,(size(FE(366:end,:),1)*24))'.^2);
mse_multivaraite_sarima = mean(reshape((FE(366:end,:)-forecast_summary_mv_sarima(:,:,1)),1,(size(FE(366:end,:),1)*24))'.^2);
mse_multivaraite_varm = mean(reshape((FE(366:end,:)-forecast_summary_varm(:,:,1)),1,(size(FE(366:end,:),1)*24))'.^2);


% compute MSE for every single hour and the different seasons separately in order to check the effects from
% Paper Ziel & Weron  
Error_univariate = (TT(8761:end,:).ForecastingError-forecast_summary(:,:).ErrorForecast)'; 
mse_univariate_per_hour = mean((reshape(Error_univariate,24,(size(Error_univariate,2)/24)))'.^2);
mse_multivariate_mv_per_hour = mean(FE(366:end,:).^2);
mse_multivaraite_cdm_per_hour = mean(((FE(366:end,:)-forecast_summary_mv(:,:,1)).^2));
mse_multivaraite_sarima_per_hour = mean(((FE(366:end,:)-forecast_summary_mv_sarima(:,:,1)).^2));
mse_multivaraite_varm_per_hour = mean(((FE(366:end,:)-forecast_summary_varm(:,:,1)).^2));


figure(figure_nbr)
plot(TT(8761:8761+23,:).Time, mse_multivariate_mv_per_hour)
hold on
plot(TT(8761:8761+23,:).Time, mse_univariate_per_hour)
hold on 
plot(TT(8761:8761+23,:).Time, mse_multivaraite_cdm_per_hour)
hold on 
plot(TT(8761:8761+23,:).Time, mse_multivaraite_sarima_per_hour)
hold on 
plot(TT(8761:8761+23,:).Time, mse_multivaraite_varm_per_hour)
datetick('x','HH')
xlabel('Hour of day')
ylabel('Error')
set(gca,'FontSize',14)
legend('Forecasting Error','Error univariate cmd','Error multivariate cmd','Error multivariate sarima','Error multivariate varm','Location','northeastoutside')
figure_nbr = figure_nbr + 1; 

%% Errors per Time period

% *** daily errors ***
% univariate 
TT_error_uv = timetable(TT(8761:end,:).ForecastingError-forecast_summary(:,:).ErrorForecast,'TimeStep',time_steps,'StartTime',TT(8761,:).Time, 'VariableNames',{'ErrorOfError'});
code_day_error_uv = code_day(8761:end,:);
[coding_day_error_uv, ia, idx_day_error_uv] = unique(code_day_error_uv);
mean_error_day_uv = accumarray(idx_day_error_uv, TT_error_uv.ErrorOfError,[],@mse);
% multivariate cmd 
test = FE(366:end,:)-forecast_summary_mv(:,:,1);
TT_error_mv_cmd = reshape((FE(366:end,:)-forecast_summary_mv(:,:,1)),1,(size(FE(366:end,:),1)*24))';
mean_error_day_mv_cmd = accumarray(idx_day_error_uv, TT_error_mv_cmd,[],@mse);
% multivariate SARIMA 
TT_error_mv_sarima = reshape((FE(366:end,:)-forecast_summary_mv_sarima(:,:,1)),1,(size(FE(366:end,:),1)*24))';
mean_error_day_mv_sarima = accumarray(idx_day_error_uv, TT_error_mv_sarima,[],@mse);
% VAR Model  
TT_error_mv_varm = reshape((FE(366:end,:)-forecast_summary_varm(:,:,1)),1,(size(FE(366:end,:),1)*24))';
mean_error_day_mv_varm = accumarray(idx_day_error_uv, TT_error_mv_varm,[],@mse);

% Plot the errors against each other over time 

[years_error,ia,idx_years] = unique(year(TT_error_uv.Time));
figure(figure_nbr)
tiledlayout(size(years_error,1),1)
for i = 1:size(years_error,1)
    nexttile
    plot(TT_error_uv(year(TT_error_uv.Time) == years_error(i,:),:).Time,TT_error_uv(year(TT_error_uv.Time) == years_error(i,:),:).ErrorOfError)
    hold on 
    plot(TT_error_uv(year(TT_error_uv.Time) == years_error(i,:),:).Time,TT_error_mv_cmd(year(TT_error_uv.Time) == years_error(i,:),:))
    hold on 
    plot(TT_error_uv(year(TT_error_uv.Time) == years_error(i,:),:).Time,TT_error_mv_sarima(year(TT_error_uv.Time) == years_error(i,:),:))
    hold on 
    plot(TT_error_uv(year(TT_error_uv.Time) == years_error(i,:),:).Time,TT_error_mv_varm(year(TT_error_uv.Time) == years_error(i,:),:))    
    xlabel('Time')
    ylabel('Error')
    set(gca,'FontSize',14)
    legend('Error univariate cmd','Error multivariate cmd','Error multivariate sarima','Error multivariate varm','Location','northeastoutside')
end
figure_nbr = figure_nbr + 1;

% plot daily mse against each other over time 
figure(figure_nbr)
tiledlayout(size(years_error,1),1)
for i = 1:size(years_error,1)
    nexttile
    plot(timevec_all_forecasts_mv(year(timevec_all_forecasts_mv) == years_error(i,:),:),mean_error_day_uv(year(timevec_all_forecasts_mv) == years_error(i,:),:))
    hold on 
    plot(timevec_all_forecasts_mv(year(timevec_all_forecasts_mv) == years_error(i,:),:),mean_error_day_mv_cmd(year(timevec_all_forecasts_mv) == years_error(i,:),:))
    hold on 
    plot(timevec_all_forecasts_mv(year(timevec_all_forecasts_mv) == years_error(i,:),:),mean_error_day_mv_sarima(year(timevec_all_forecasts_mv) == years_error(i,:),:))
    hold on 
    plot(timevec_all_forecasts_mv(year(timevec_all_forecasts_mv) == years_error(i,:),:),mean_error_day_mv_varm(year(timevec_all_forecasts_mv) == years_error(i,:),:))
    xlabel('Time')
    ylabel('Error')
    set(gca,'FontSize',14)
    legend('MSE univariate cmd','MSE multivariate cmd','MSE multivariate sarima','MSE multivariate varm','Location','northeastoutside')
end
figure_nbr = figure_nbr + 1;

% *** monthly errors ***
% univaraite 
code_month_error_uv = (code_day(8761:end,:) - mod(code_day(8761:end,:),100))/100;
[coding_month_error_uv, ia, idx_month_error_uv] = unique(code_month_error_uv);
mean_error_month_uv = accumarray(idx_month_error_uv, TT_error_uv.ErrorOfError,[],@mse);
mean_error_month_mv_cmd = accumarray(idx_month_error_uv, TT_error_mv_cmd,[],@mse);
mean_error_month_mv_sarima = accumarray(idx_month_error_uv, TT_error_mv_sarima,[],@mse);
mean_error_month_varm = accumarray(idx_month_error_uv, TT_error_mv_varm,[],@mse);


time_months = (start_date_forecast:calmonths(1):(start_date_forecast + calyears(size(years_error,1))-days(1)))';

% plot daily mse against each other over time 
figure(figure_nbr)
tiledlayout(size(years_error,1),1)
for i = 1:size(years_error,1)
    nexttile
    plot(time_months(year(time_months) == years_error(i,:),:),mean_error_month_uv(year(time_months) == years_error(i,:),:))
    hold on 
    plot(time_months(year(time_months) == years_error(i,:),:),mean_error_month_mv_cmd(year(time_months) == years_error(i,:),:))
    hold on 
    plot(time_months(year(time_months) == years_error(i,:),:),mean_error_month_mv_sarima(year(time_months) == years_error(i,:),:))
    hold on 
    plot(time_months(year(time_months) == years_error(i,:),:),mean_error_month_varm(year(time_months) == years_error(i,:),:))
    xlabel('Time')
    ylabel('Error')
    set(gca,'FontSize',14)
    legend('MSE univariate cmd','MSE multivariate cmd','MSE multivariate sarima','MSE multivariate varm','Location','northeastoutside')
end
figure_nbr = figure_nbr + 1;


%% Stand 19.11.20
%% Absolute and relative evaluiation of the forecasts 

% **** Absolute evaluation of forecasts: Diebold Mariano Test ****

% Regression of Y on Y_hat 
lm = fitlm(forecast_summary(:,:).ErrorForecast, TT(8761:end,:).ForecastingError);
[EstCov,se,coeff] = hac(lm, 'type','HAC'); % since we deal with heterosked. and autocorrelation need robust SE estimation
% waldtest for coefficient estimates 
p_val_cmd_uv = waldtest(lm,EstCov);
% if p-Value is high we have no significant result which means that we
% cannot reject the H0 that intercept = 0, slope = 1 -> looks good! 

% **** Relative evaluation of forecasts: Murphy Diagram ****
% Used fct from
% https://de.mathworks.com/matlabcentral/fileexchange/60974-murphy-diagrams
% (Reproduction of R Code from Krueger) 
% murphydiagram(f1, f2, y, varargin) where f1 = forecast1; f2 = forecast2,
% y = realizations
% exemplary here for hour 0 (1.row of data) 
% idea: transform data back in one timeseries 
forecast_summary_mv_ts = (reshape(forecast_summary_mv(:,:,1),1, size(forecast_summary_mv,1)*24))';
labels = {'CDM_MV', 'SARIMA_MV'};
[Stheta, theta] = murphydiagram(forecast_summary_mv(2:size(forecast_summary_mv,1)-1,1,1), forecast_summary_mv_sarima(2:size(forecast_summary_mv_sarima,1)-1,1,1), FE(367:end,1), 'labels',labels)


%% Functions 

function [mse] = mse(X)
    mse = mean(X.^2);
end


% monthly structure 
function [coding_every_month,mean_every_month,coding_mean_month,mean_month]  = myfun_mean_every_month(Data)
    
    coding_months = ((year(Data.Time))-2000) * 100 + month(Data.Time);
    [coding_every_month, ia, idx_every_month] = unique(coding_months);
    mean_every_month = accumarray(idx_every_month, Data.ForecastingError,[],@mean);
    
    [coding_mean_month, ia, idx_mean_month] = unique(month(Data.Time));
    mean_month = accumarray(idx_mean_month, Data.ForecastingError,[],@mean);   
end

function myfun_plot_mean_yearly(mean_every_month, mean_month, coding_month, figure_nbr)
    
    nbr_of_years = (size(mean_every_month,1)/12);
    mean_every_month = reshape(mean_every_month,12,nbr_of_years);
    figure(figure_nbr)
    plot(coding_month, mean_month,'LineWidth',3)
    xlim([coding_month(1,:) coding_month(end,:)])
    xlabel('Month of the year')
    ylabel('Mean of the price forecasting error [Euro per MWh]')
    for i = 1:nbr_of_years
        hold on 
        plot(coding_month,mean_every_month(:,i),'LineWidth',1.5)
    end
    legend('Mean','2015','2016','2017','2018')
    set(gca,'FontSize',18)
    title('Mean of price forecasting errors for every month')
    hold off
end

% weekly structure

function [mean_weekday, coding_weekday] = myfun_mean_of_weekday(Data,weekday_code)

    coding_weekday_month_year = (year(Data.Time)-2000) * 1000 + month(Data.Time) * 10 + weekday_code;
    [coding_weekday,ia ,idx_weekday] = unique(coding_weekday_month_year);
    mean_weekday = accumarray(idx_weekday, Data.ForecastingError,[],@mean);
end

function myfun_plot_mean_weekday(mean_weekday, coding_weekday,figure_nbr)
counter = 1;
mat_months = ["Jan";"Feb";"Mar";"Apr";"May";"Jun";"Jul"; "Aug";"Sep";"Oct";"Nov";"Dec"]; 
figure(figure_nbr)
tiledlayout(4,3)
for j = 1:12
    nexttile
    for i = counter:84:size(mean_weekday,1) %84 = 7 days * 12 months per year
        plot(mean_weekday(i:i+6,:),'LineWidth',1.3)
        xlabel('Day of ' + mat_months(j,:) + ' (1 = Sun; 2 = Mon; ...)')
        ylabel('Mean forecasting error')
        legend('2015','2016','2017','2018', 'Location','northeastoutside')
        hold on    
    end
    hold off
    counter = counter + 7;
end
end

% daily structure 

function [mean_hourly_weekday,coding_hours, coding_hour_vec] = myfun_mean_hourly_per_weekday(Data_adj_weekly_seasonality)

coding_hour_vec = weekday(Data_adj_weekly_seasonality.Time)*100 + hour(Data_adj_weekly_seasonality.Time);
[coding_hours,ia ,idx_hour] = unique(coding_hour_vec);
mean_hourly_weekday = accumarray(idx_hour, Data_adj_weekly_seasonality.ForecastingError,[],@mean);
end

function myfun_plot_hourly_mean(mean_hourly_weekday,figure_nbr)
figure(figure_nbr)
tiledlayout(4,2)
day = 1;
mat_weekdays = ["Sun";"Mon";"Tue";"Wed";"Thu";"Fri";"Sat"];
for i = 1:24:167
    nexttile 
    plot(mean_hourly_weekday(i:i+23,:),'LineWidth',1.3)
    xlabel('Hour on ' + mat_weekdays(day,:))
    ylabel('Mean forecasting error')     
    day = day + 1;
end
end

function [p_val] = waldtest(lm,EstCov)
    r = [0;1];
    theta = lm.Coefficients.Estimate;
    wald = (theta-r)' * inv(EstCov) * (theta-r); 
    p_val = (1-chi2cdf(wald,2));
end