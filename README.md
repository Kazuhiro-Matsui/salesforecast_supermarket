# スーパーマーケット惣菜売上予測モデル：LightGBM による日次需要予測の入門デモ

> **TL;DR** — Kaggle "Store Sales - Time Series Forecasting" のデータから、**特定店舗（店舗番号 3）の惣菜（PREPARED FOODS）日次売上**を LightGBM で予測する入門デモ。
> 2016-07-15 〜 2017-07-31 のデータで学習し、**2017-08-01 〜 2017-08-15 の 15 日間**の売上を予測。
> 惣菜の発注量決定に使える精度かどうかを定性的に評価した。
> 機械学習の学習を始めて間もない頃に作成した個人開発であり、後続のプロジェクト（心血管疾患予測 / 予知保全 / タクシー需要予測）に比べると分析の深さや統計的検証は限定的だが、**「需要予測ワークフローの全体像を最後まで通す」**という最初の到達点として位置付けている。

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Kazuhiro-Matsui/salesforecast_supermarket/blob/main/demo.ipynb)

---

## 📌 この成果物で示せること（採用担当者向け要約）

| 観点              | 内容                                                                                          |
| ----------------- | --------------------------------------------------------------------------------------------- |
| **位置付け**      | **機械学習学習初期の入門デモ**（最初に「一通り通せた」プロジェクト）                          |
| **ドメイン**      | 小売・スーパーマーケット・需要予測・発注最適化                                                |
| **手法選定**      | 表形式データの定番である **LightGBM** で日次売上を回帰予測                                    |
| **技術スタック**  | Python 3.x / pandas / LightGBM / scikit-learn / matplotlib / Google Colab                     |
| **学んだこと**    | データ前処理、特徴量エンジニアリング、複数オープンデータの結合、時系列の train/test 分割     |

> ⚠️ **正直な注記**：本プロジェクトは機械学習の勉強を始めた初期に作成したもので、ハイパーパラメータ調整、交差検証、誤差の統計的評価、特徴量重要度の深掘りといった工程は最小限です。同じドメインで再挑戦するなら何をどう改善するかは、後述「**7. 反省と次に作るならどうするか**」にまとめています。

---

## 1. 背景：なぜこの課題を選んだか

スーパーマーケットの惣菜は**当日中に売り切ることが前提の生鮮品**であり、需要を読み違えると次の 2 つの損失が同時に発生する:

- **発注過多** → 値引き販売・廃棄ロス（フードロスの大きな原因）
- **発注過少** → 機会損失・棚の空き・顧客満足度の低下

日々の発注を経験と勘で決めている店舗は今も多く、これを「**機械学習で当日の需要を事前に当てられないか**」というのが本プロジェクトの動機である。

機械学習を学び始めた段階で、まず「**実データを使って前処理から評価まで一通り経験する**」ことを目的に、Kaggle の公開データセットを題材に選んだ。

---

## 2. データ

- **Source**: [Kaggle - Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)
- **対象店舗**: `store_nbr = 3`（1 店舗に絞り込み）
- **対象カテゴリ**: `PREPARED FOODS`（惣菜）
- **学習期間**: 2016-07-15 〜 2017-07-31（約 1 年分、396 日）
- **予測期間**: 2017-08-01 〜 2017-08-15（**15 日間**）

参考用に、可視化のためだけに DAIRY（乳製品）/ BREAD-BAKERY（パン類）/ PRODUCE（青果）の 3 カテゴリも読み込んでいる。

### 使用した特徴量

| 特徴量          | 内容                                                        | データソース                |
| --------------- | ----------------------------------------------------------- | --------------------------- |
| `onpromotion`   | その日にプロモーション対象だったアイテム数                  | `train.csv` (生データ)      |
| `is_event`      | 特別イベント日かどうか（0/1）                               | `holidays_events.csv`       |
| `is_workday`    | 出勤日 / 休日（土日祝＋振替休日 / 振替出勤日まで考慮）      | `holidays_events.csv` + 曜日 |
| `day_of_week`   | 曜日（月=0, ..., 日=6）                                     | 日付列から生成              |
| `oil_price`     | 原油価格（景気指標として欠損値は ffill / bfill で補完）     | `oil.csv`                   |

### 前処理ハイライト

- 休日判定では **`transferred` フラグ**を見て「振り替えられた祝日」を除外し、`'Work Day'` タイプの日は土日でも出勤日扱いに上書き（→ エクアドルの祝日制度に踏み込んで処理）
- 原油価格は土日・祝日に欠損するため、`ffill().bfill()` で前後の平日価格を継承

---

## 3. 手法：LightGBM による回帰

学習初期のプロジェクトなので、モデル選定は「**まず動かして全体像を掴む**」ことを優先し、表形式データの定番である LightGBM を採用した。

### モデルの位置付け

| 選択肢                       | 採用 | 当時の理由                                                          |
| ---------------------------- | :--: | ------------------------------------------------------------------- |
| 線形回帰                     |  ❌  | 曜日・休日効果を綺麗に表現できないと聞いていた                      |
| **LightGBM**                 |  ✅  | 表形式データに強く、Kaggle で実績がある定番モデル                   |
| SARIMA / Prophet (時系列専用) |  ❌  | 当時はまだ時系列モデルの理解が浅かった（→ 反省ポイント）             |
| ニューラルネット             |  ❌  | データ量が少なすぎる（396 日のみ）                                  |

### 検証戦略

時系列データのため、**期間で train / test を分割**:

- Train: 2016-07-15 〜 2017-07-31
- Test : 2017-08-01 〜 2017-08-15（15 日間）

---

## 4. 結果

Notebook 内で予測結果と実測値を時系列でプロットして比較。**「発注量決定に使えるレベルか」を定性的に評価**している。

> ⚠️ MAE / RMSE / MAPE といった定量指標を表として整理する作業は当時行っていません。
> 詳細な数値は Notebook を直接実行して確認するか、「7. 反省と次に作るならどうするか」を参照してください。

### Notebook で確認できる出力

- 4 カテゴリ（DAIRY / BREAD/BAKERY / PREPARED FOODS / PRODUCE）の売上時系列推移プロット
- 惣菜カテゴリの予測値 vs 実測値の重ね描き
- 特徴量の summary（イベント日 12 件、休日 588 件、平日 996 件）

---

## 5. このプロジェクトで学んだこと

| 学習項目                   | 具体的に身についたこと                                                    |
| -------------------------- | ------------------------------------------------------------------------- |
| **pandas でのデータ整形**   | ZIP 解凍、複数 CSV の join、日付型変換、インデックス操作                    |
| **特徴量エンジニアリング** | 曜日・休日・イベントを 0/1 フラグへ変換、外部データ（原油価格）の結合     |
| **欠損値処理**             | 土日に欠損する時系列データを `ffill().bfill()` で補完                     |
| **ドメイン知識の活用**     | エクアドルの `transferred` 祝日や `Work Day`（振替出勤日）の取扱い         |
| **LightGBM の基本**        | 学習・推論 API、回帰タスクの流れ                                          |
| **時系列分割**             | 「ランダム CV ではなく期間で切る」という時系列予測の鉄則                  |

---

## 6. 限界とバイアス（正直なふりかえり）

| 限界点                                          | 内容                                                                                       |
| ----------------------------------------------- | ------------------------------------------------------------------------------------------ |
| **検証期間が 1 ホールドアウトのみ**             | TimeSeriesSplit による複数 fold 評価をしていない → 結果のばらつきが見えない                |
| **誤差の定量指標が不十分**                      | MAE / RMSE / MAPE を計算・記録していない → 「使えるかどうか」が定性判断にとどまる         |
| **ハイパーパラメータがデフォルト**              | `num_leaves`, `learning_rate`, `n_estimators` 等を調整していない                            |
| **ラグ特徴量・移動平均がない**                  | 「前週同曜日の売上」「直近 7 日平均」など、時系列予測の定石を入れていない                  |
| **1 店舗 1 カテゴリのみ**                       | 汎用性の検証になっていない（他店舗・他カテゴリで動かしていない）                          |
| **特徴量重要度の分析がない**                    | `lgb.plot_importance()` での寄与度確認をしていない                                         |
| **予測区間（不確実性）がない**                  | 点予測のみで「どれくらい外れ得るか」が出ない（→ 発注業務では本来必須）                    |

---

## 7. 反省と次に作るならどうするか

このプロジェクトを今の知識でやり直すなら、以下を導入する:

### 7.1 検証の厳密化
- **TimeSeriesSplit** による複数 fold 評価
- **MAE / RMSE / MAPE / sMAPE** を fold ごとに記録し、平均±標準偏差で報告
- ベースライン（前週同曜日の値をそのまま予測）との比較

### 7.2 時系列らしい特徴量
- **ラグ特徴量**（7 日前・14 日前・28 日前の売上）
- **移動平均**（直近 7 日平均、直近 28 日平均）
- 曜日 × 月の交互作用、月初・月末フラグ

### 7.3 モデルの拡張
- **LightGBM の Quantile Regression** で予測区間（10% / 50% / 90% 分位点）を出す
- **Prophet** や **SARIMAX** との比較
- 後続プロジェクト（タクシー需要予測）で学んだ **GAM/BAM による解釈可能モデル**との比較

### 7.4 業務観点の評価
- 「予測値で発注したらどれだけ廃棄ロス・機会損失が出るか」のシミュレーション
- 過剰発注と過少発注で**コストが非対称**であることを反映した非対称損失関数

これらは後続のプロジェクト（[Cardiovascular-Disease](https://github.com/Kazuhiro-Matsui/Cardiovascular-Disease)、[Predictive-Maintenance-Dataset](https://github.com/Kazuhiro-Matsui/Predictive-Maintenance-Dataset)、[ride-hailing](https://github.com/Kazuhiro-Matsui/ride-hailing)）で実際に取り入れており、**初期から現在までの学習の伸び**を見ていただける材料にしていただければ幸いです。

---

## 📁 ファイル構成

```
salesforecast_supermarket/
├── README.md
└── demo.ipynb   # データ取得 → 前処理 → 特徴量作成 → LightGBM 学習 → 予測 → 可視化
```

---

## 🚀 実行方法

### 1. Google Colab で開く（推奨）

ノートブック冒頭の「Open In Colab」バッジから直接 Colab で開けます。

<https://colab.research.google.com/github/Kazuhiro-Matsui/salesforecast_supermarket/blob/main/demo.ipynb>

### 2. データの準備

1. Kaggle の [Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting) から `store-sales-time-series-forecasting.zip` をダウンロード
2. Google Drive の `MyDrive/salesforecast_supermarket_demo/` 配下に zip ファイルを配置
   - Notebook 内のパス: `/content/drive/MyDrive/salesforecast_supermarket_demo/store-sales-time-series-forecasting.zip`
3. Notebook のセルで Google Drive をマウントすると、`/content/dataset/` に自動展開されます

> 💡 別の場所に zip を置いた場合は、Notebook 内の `unzip` コマンドのパスを編集してください。

### 3. ノートブックを上から順に実行

セルを順に実行すると、以下が自動で完了します。

1. Google Drive のマウント
2. ZIP 解凍（`train.csv`, `holidays_events.csv`, `oil.csv` 等を取得）
3. 店舗 3・対象 4 カテゴリへのフィルタリング
4. 休日・イベント・原油価格などの特徴量を結合
5. 各カテゴリの売上推移可視化
6. LightGBM での学習と 15 日間の予測
7. 予測値 vs 実測値のプロット

Colab 無料枠の CPU で**数秒程度**で完走します。

---

## 👤 Author

**Kazuhiro Matsui**
GitHub: [@Kazuhiro-Matsui](https://github.com/Kazuhiro-Matsui)

---

## 🔗 関連プロジェクト（このリポジトリ以降の学習成果）

- **[Cardiovascular-Disease](https://github.com/Kazuhiro-Matsui/Cardiovascular-Disease)** — GAM/BAM による解釈可能な医療スクリーニング AI
- **[Predictive-Maintenance-Dataset](https://github.com/Kazuhiro-Matsui/Predictive-Maintenance-Dataset)** — ベイズロジスティック回帰による予知保全 AI
- **[ride-hailing](https://github.com/Kazuhiro-Matsui/ride-hailing)** — GAM/BAM による時空間タクシー需要予測
