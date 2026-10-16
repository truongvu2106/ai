# Probability & Statistics for AI — Python Practice

**AI & ML — Learn • Build • Grow**

Một repository Python thực hành các kiến thức trong **Module 2.3 — Probability & Statistics for AI**. Mục tiêu là giúp người học nhìn thấy toán học hoạt động trong code: từ xác suất và Bayes' theorem đến Maximum Likelihood Estimation, Standard Error, Confidence Interval và classification metrics.

> Phù hợp cho người mới học Python cho AI/ML. Code ưu tiên rõ ràng, hàm nhỏ, type hints, docstrings và kiểm thử tự động.

## Nội dung

| Topic | Nội dung thực hành |
|---|---|
| Probability fundamentals | Union rule, conditional probability, independence, Bayes' theorem |
| Expectation, Variance, Standard Deviation | PMF có trọng số; population và sample variance |
| Gaussian Distribution | Gaussian PDF và Z-score |
| Central Limit Theorem | Mô phỏng sampling distribution của sample means |
| Maximum Likelihood Estimation | Ước lượng xác suất thành công của Bernoulli/coin toss |
| Sampling & Statistical Inference | Random vs convenience sampling, sample proportion, Standard Error, Margin of Error, Wald/Wilson Confidence Intervals, z-test |
| Classification & Evaluation | Confusion Matrix, Accuracy, Precision, Recall, Specificity, F1, Log Loss, ROC-AUC, calibration table, ECE |

## Yêu cầu

- Python 3.10 trở lên
- `numpy`
- `matplotlib`
- `pytest` để chạy tests

## Cài đặt

### 1. Clone repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd probability-statistics-for-ai
```

Thay `<YOUR_GITHUB_REPOSITORY_URL>` bằng URL repository sau khi bạn tạo repository trên GitHub.

### 2. Tạo và kích hoạt virtual environment

Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Cài dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## Chạy bài thực hành

Từ thư mục gốc repository:

```bash
python examples/run_practice.py
```

Script sẽ chạy các ví dụ và tạo ba biểu đồ trong `outputs/`:

- `sampling_bias_demo.png` — so sánh tỷ lệ AI-use của toàn bộ population, random sample và convenience sample.
- `clt_sampling_means.png` — so sánh population lệch phải với sampling distributions của sample means.
- `confidence_interval.png` — hiển thị sample proportion và khoảng tin cậy 95%.

Các ví dụ chính dùng cùng những con số đã trình bày trong series để dễ đối chiếu:

- Bayes: prior spam = 10%, likelihoods = 80% và 20%.
- Expectation: `X = [0, 2, 4]`, `P = [0.25, 0.50, 0.25]`.
- Sampling bias: population có tỷ lệ AI-use chính xác 50%; convenience sample chỉ lấy từ tech enthusiasts sẽ cho tỷ lệ cao hơn.
- MLE: 7 heads trong 10 lần tung đồng xu.
- Inference: 240/400 người sử dụng AI.
- Classification: 100 email, gồm 20 spam và 80 non-spam; TP=15, FN=5, FP=10, TN=70. Có thêm Log Loss, ROC-AUC và Expected Calibration Error (ECE).

## Chạy tests

```bash
pytest -q
```

Các tests kiểm tra công thức, xử lý dữ liệu đầu vào và kết quả số học quan trọng.

## Một số kết quả tham khảo

Ví dụ `240/400`:

- Sample proportion: `0.60`
- Plug-in Standard Error: khoảng `0.0245` (2.45 percentage points)
- Margin of Error cho Wald 95% CI: khoảng `0.0480` (4.80 percentage points)
- Wald 95% Confidence Interval: khoảng `[0.552, 0.648]`
- One-sample z-test với `H0: p=0.50`: `z=4.0`, two-sided p-value khoảng `0.000063`

**Lưu ý:** Wald interval được triển khai để khớp với ví dụ giảng dạy `p̂ ± z × SE`. Với cỡ mẫu nhỏ hoặc tỷ lệ gần 0/1, Wald interval có thể hoạt động kém; repository cũng cung cấp Wilson interval để so sánh. Cả khoảng tin cậy lẫn z-test đều dựa trên các điều kiện xấp xỉ và giả định lấy mẫu phù hợp.

## Ghi chú về diễn giải thống kê

- Standard Deviation mô tả mức độ phân tán của các quan sát; Standard Error mô tả mức biến động của một thống kê qua các mẫu.
- Margin of Error trong ví dụ là nửa độ rộng của Wald confidence interval, không phải bảo đảm sai số thực tế không vượt quá giá trị này.
- Confidence level 95% được hiểu theo tính chất lặp lại của phương pháp: trong nhiều lần lấy mẫu phù hợp, khoảng được xây dựng bằng phương pháp này sẽ bao phủ tham số thật khoảng 95% số lần.
- p-value không phải xác suất giả thuyết null là đúng.
- ROC-AUC đo khả năng xếp hạng, không đo probability calibration. ECE là thước đo calibration phụ thuộc số bins, không nên dùng như một đánh giá duy nhất.
- CLT nói về sampling distribution của sample mean trong những điều kiện phù hợp; không nói dữ liệu gốc tự động trở thành Gaussian.

## Cấu trúc repository

```text
probability-statistics-for-ai/
├── src/probability_stats_ai/
│   ├── probability.py       # Probability rules and Bayes' theorem
│   ├── descriptive.py       # Expectation, variance, standard deviation
│   ├── distributions.py     # Gaussian PDF, Z-score, CLT simulation
│   ├── estimation.py        # Bernoulli Maximum Likelihood Estimation
│   ├── inference.py         # SE, ME, confidence intervals, proportion z-test
│   └── classification.py    # Binary metrics, cross-entropy, ROC-AUC, calibration
├── examples/
│   └── run_practice.py     # End-to-end examples and visualizations
├── exercises/
│   └── CHALLENGES.md       # Extra coding challenges
├── tests/                  # Unit tests
├── outputs/                # Generated figures
├── requirements.txt
├── pyproject.toml
├── LICENSE
├── VIDEO_PROMPTS.md
└── README.md
```

## Gợi ý lộ trình học

1. Đọc một module tại `src/probability_stats_ai/`.
2. Chạy `examples/run_practice.py` và đối chiếu output.
3. Thử thay đổi dữ liệu đầu vào, quan sát sự thay đổi của kết quả.
4. Làm bài trong `exercises/CHALLENGES.md`.
5. Chạy `pytest -q` sau khi sửa code.

## Đóng góp

Issues và pull requests được hoan nghênh. Khi thêm tính năng, hãy bổ sung test, giải thích giả định thống kê và giữ ví dụ dễ chạy cho người mới.

## Disclaimer

Repository phục vụ học tập, không thay thế việc kiểm định thống kê chuyên sâu hoặc tư vấn chuyên môn cho quyết định có rủi ro cao.
