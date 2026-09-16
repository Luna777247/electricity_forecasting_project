# electricity_forecasting_project
HƯỚNG DẪN CHẠY PIPELINE DỰ BÁO ĐIỆN NĂNG

1. Cài thư viện:
   pip install -r requirements.txt

2. Chạy benchmark tái lập kết quả exploratory:
   python electricity_forecasting_pipeline.py --input "household_power_consumption - Copy.csv" --output-dir outputs --start-date 2007-01-01

3. Chạy chế độ forecast nghiêm ngặt (không dùng các covariate điện tại cùng timestamp tương lai):
   python electricity_forecasting_pipeline.py --input "household_power_consumption - Copy.csv" --output-dir outputs_strict --start-date 2007-01-01 --strict-forecast

Pipeline gồm:
- khôi phục date/datetime
- fill missing theo gap
- resample 15 phút
- time/lag/rolling features
- split 70/15/15 theo thời gian
- Seasonal Naive
- SARIMA
- XGBoost
- LSTM 60 phút / 6 giờ / 24 giờ
- weighted hybrid
- XGBoost + Residual LSTM
- MAE/RMSE/MAPE/sMAPE/WAPE

Lưu ý: kết quả trong báo cáo là benchmark exploratory đã chạy trong phiên làm việc. Khi bật --strict-forecast, metric có thể thay đổi vì loại các biến đồng thời chưa chắc biết trước ở thời điểm dự báo.
