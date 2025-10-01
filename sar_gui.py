from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont, QImage, QIntValidator, QDoubleValidator, QPixmap
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QMessageBox, 
    QComboBox, QPushButton, QFileDialog, QListWidget, QLabel, QGridLayout,
    QFrame, QLineEdit, QTextEdit, QProgressDialog
)
import sys
import pandas as pd
import sarlib

class Worker(QThread):
    finished = Signal()

    def __init__(self, fcn):
        super().__init__()
        self.fcn = fcn

    def run(self):
        self.fcn()
        self.finished.emit()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.initUI()

        self.data = {
            'data': None,
            'file': None,
            'model': None,
            'ssa': None
        }

    def initUI(self):
        title = "Statistical Agnostic Regression"
        self.setWindowTitle(title)
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Window title
        title_label = QLabel("SAR", alignment=Qt.AlignHCenter)
        title_label.setFont(QFont(title_label.font().family(), 100, QFont.Bold))
        subtitle_label = QLabel(title, alignment=Qt.AlignHCenter)
        subtitle_label.setFont(QFont(title_label.font().family(), 25))
        layout.addWidget(title_label)
        layout.addWidget(subtitle_label)

        # Frame for data file
        frm_file = QFrame()
        frm_file.setFrameShape(QFrame.StyledPanel)
        frm_file.setFrameShadow(QFrame.Plain)
        lyt_file = QHBoxLayout(frm_file)
        self.edt_file = QLineEdit(placeholderText="No file loaded", readOnly=True)
        btn_load = QPushButton("Load CSV file")
        btn_load.clicked.connect(self.load_data)
        btn_load.setToolTip("Load a CSV file with the data to analyze")
        lyt_file.addWidget(QLabel("File:"))
        lyt_file.addWidget(self.edt_file)
        lyt_file.addWidget(btn_load)
        layout.addWidget(frm_file)

        # Frame for data and parameters
        frm_data = QFrame()
        frm_data.setFrameShape(QFrame.StyledPanel)
        frm_data.setFrameShadow(QFrame.Plain)
        lyt_data = QGridLayout(frm_data)
        lyt_data.addWidget(QLabel("Response:"), 0, 0, 1, 1)
        lyt_data.addWidget(QLabel("Predictors:"), 0, 1, 1, 1)
        self.lst_response = QListWidget(selectionMode=QListWidget.SingleSelection)
        self.lst_predictors = QListWidget(selectionMode=QListWidget.MultiSelection)
        lyt_data.addWidget(self.lst_response, 1, 0, 1, 1)
        lyt_data.addWidget(self.lst_predictors, 1, 1, 1, 1)
        btn_scatter = QPushButton("Show scatter plots")
        btn_scatter.clicked.connect(self.show_scatter)
        btn_scatter.setToolTip("Show scatter plots of the response vs predictors")
        lyt_data.addWidget(btn_scatter, 2, 0, 1, 2)
        layout.addWidget(frm_data)

        # Parameters for SAR analysis
        lyt_data.addWidget(QLabel("Parameters:"), 0, 2, 1, 1)
        frm_params = QFrame()
        frm_params.setFrameShape(QFrame.StyledPanel)
        frm_params.setFrameShadow(QFrame.Plain)
        lyt_params = QGridLayout(frm_params)
        lyt_data.addWidget(frm_params, 1, 2, 1, 1)

        lyt_params.addWidget(QLabel("Realizations:"), 0, 0, 1, 1)
        self.edt_realiz = QLineEdit("100", alignment=Qt.AlignRight)
        self.edt_realiz.setToolTip("Number of realizations (sampling) for the analysis")
        self.edt_realiz.setValidator(QIntValidator(1, 10000, self))
        lyt_params.addWidget(self.edt_realiz, 0, 1, 1, 1)

        lyt_params.addWidget(QLabel("Norm:"), 1, 0, 1, 1)
        self.cmb_norm = QComboBox()
        self.cmb_norm.addItems(['rmse', 'epsins'])
        self.cmb_norm.setToolTip("Norm for the loss function: RMSE or Epsilon-insensitive")
        lyt_params.addWidget(self.cmb_norm, 1, 1, 1, 1)

        lyt_params.addWidget(QLabel("Training mode:"), 2, 0, 1, 1)
        self.cmb_mode = QComboBox()
        self.cmb_mode.addItems(['resusb', 'kfold', 'leaveoo'])
        self.cmb_mode.setToolTip("Training mode: Resusbtitution, K-Fold, or Leave-One-Out")
        lyt_params.addWidget(self.cmb_mode, 2, 1, 1, 1)

        lyt_params.addWidget(QLabel("Threshold eps value:"), 3, 0, 1, 1)
        self.edt_eps_0 = QLineEdit("", alignment=Qt.AlignRight)
        self.edt_eps_0.setToolTip("Value for epsilon-insensitive loss free parameter in threshold definition (leave empty for default)")
        self.edt_eps_0.setValidator(QDoubleValidator(0.0, 100.0, 2, self))
        lyt_params.addWidget(self.edt_eps_0, 3, 1, 1, 1)

        lyt_params.addWidget(QLabel("Alpha:"), 4, 0, 1, 1)
        self.edt_alpha = QLineEdit("0.05", alignment=Qt.AlignRight)
        self.edt_alpha.setToolTip("Significance level for the classical analysis")
        self.edt_alpha.setValidator(QDoubleValidator(0.0, 1.0, 2, self))
        lyt_params.addWidget(self.edt_alpha, 4, 1, 1, 1)

        lyt_params.addWidget(QLabel("Upper Bound:"), 5, 0, 1, 1)
        self.cmb_bound = QComboBox()
        self.cmb_bound.addItems(['pacbayes', 'vapnik', 'igp', 'igp_approx'])
        self.cmb_bound.setToolTip("Upper Bound for the analysis: PAC-Bayes, Vapnik, IGP, or IGP approximation")
        self.cmb_bound.currentTextChanged.connect(self.enable_bound_params)
        lyt_params.addWidget(self.cmb_bound, 5, 1, 1, 1)

        lyt_params.addWidget(QLabel("Eta:"), 6, 0, 1, 1)
        self.edt_eta = QLineEdit("0.50", alignment=Qt.AlignRight)
        self.edt_eta.setToolTip("Eta parameter for bound calculation")
        self.edt_eta.setValidator(QDoubleValidator(0.0, 1.0, 2, self))
        lyt_params.addWidget(self.edt_eta, 6, 1, 1, 1)

        lyt_params.addWidget(QLabel("Dropout rate:"), 7, 0, 1, 1)
        self.edt_dropout = QLineEdit("0.50", alignment=Qt.AlignRight)
        self.edt_dropout.setToolTip("Dropout rate for bound calculation in PAC-Bayes Upper Bound")
        self.edt_dropout.setValidator(QDoubleValidator(0.0, 1.0, 2, self))
        lyt_params.addWidget(self.edt_dropout, 7, 1, 1, 1)

        btn_analize = QPushButton("Run analysis")
        btn_analize.clicked.connect(self.analize)
        btn_analize.setToolTip("Run SAR analysis with the selected parameters")
        lyt_data.addWidget(btn_analize, 2, 2, 1, 1)

        # Frame for analysis (SAR)
        frm_analysis = QFrame()
        frm_analysis.setFrameShape(QFrame.StyledPanel)
        frm_analysis.setFrameShadow(QFrame.Plain)
        layout.addWidget(frm_analysis)
        lyt_analysis = QGridLayout(frm_analysis)
        self.lbl_analysis = QLabel("Analysis results:")
        lyt_analysis.addWidget(self.lbl_analysis, 0, 0, 1, 2)

        self.txt_analysis = QTextEdit(placeholderText="Analysis output...", readOnly=True)
        lyt_analysis.addWidget(self.txt_analysis, 1, 0, 1, 1)

        # Frame para botones de Sample Size Analysis
        frm_sample = QFrame()
        frm_sample.setFrameShape(QFrame.StyledPanel)
        frm_sample.setFrameShadow(QFrame.Plain)
        lyt_sample = QVBoxLayout(frm_sample)
        lyt_sample.addWidget(QLabel("Sample Size Analysis", alignment=Qt.AlignHCenter))
        btn_plot_loss = QPushButton("Plot loss")
        btn_plot_power = QPushButton("Plot p-value")
        btn_plot_coef = QPushButton("Plot coefficients")
        btn_plot_loss.clicked.connect(lambda: self.plot_ssa('loss'))
        btn_plot_power.clicked.connect(lambda: self.plot_ssa('pvalue'))
        btn_plot_coef.clicked.connect(lambda: self.plot_ssa('coef'))
        btn_plot_loss.setToolTip("Plot the loss function")
        btn_plot_power.setToolTip("Plot the power of the analysis")
        btn_plot_coef.setToolTip("Plot the coefficients of the predictors")
        lyt_sample.addWidget(btn_plot_loss)
        lyt_sample.addWidget(btn_plot_power)
        lyt_sample.addWidget(btn_plot_coef)
        lyt_analysis.addWidget(frm_sample, 1, 1, 1, 1)

    def enable_bound_params(self):
        bound = self.cmb_bound.currentText()
        if bound == 'pacbayes':
            self.edt_eta.setEnabled(True)
            self.edt_dropout.setEnabled(True)
        else:
            #self.edt_eta.setEnabled(False)
            self.edt_dropout.setEnabled(False)

    def load_data(self):
        title = "Please select a CSV file with the data"
        path, _ = QFileDialog.getOpenFileName(self, title, ".", "CSV (*.csv)")
        if not path: return
        self.edt_file.setText(path)

        # Load CSV data
        df = pd.read_csv(path)
        cols = df.columns.tolist()
        self.lst_response.clear()
        self.lst_response.addItems(cols)
        self.lst_predictors.clear()
        self.lst_predictors.addItems(cols)

        self.data['file'] = path
        self.data['data'] = df

        if cols:
            self.lst_response.setCurrentRow(0)
        for i in range(1, len(cols)):
            self.lst_predictors.item(i).setSelected(True)

    def get_xy(self):
        # Check if response or predictors are selected
        response_items = self.lst_response.selectedItems()
        predictor_items = self.lst_predictors.selectedItems()
        if not response_items or not predictor_items:
            QMessageBox.warning(self, "Selection error", 
                        "Please select a response and at least one predictor.")
            return None, None, None, None

        # Get selected response and predictor columns
        response_col = response_items[0].text()
        predictor_cols = [item.text() for item in predictor_items]
        x = self.data['data'][predictor_cols].to_numpy()
        y = self.data['data'][response_col].to_numpy()
        return x, y, predictor_cols, response_col

    def show_scatter(self):
        x, y, predictor_cols, response_col = self.get_xy()
        if x is None or y is None: return
        sarlib.show_scatter(x, y, predictor_cols, response_col)

    def analize(self):
        self.data['x'], self.data['y'], predictor_cols, response_col = self.get_xy()
        if self.data['x'] is None or self.data['y'] is None: return
        msg = f"Analysis results for {response_col} vs {', '.join(predictor_cols)}:"
        self.lbl_analysis.setText(msg)

        # Get parameters for SAR analysis
        params = {}
        params['n_realiz'] = int(self.edt_realiz.text())
        params['norm'] = self.cmb_norm.currentText()
        params['mode'] = self.cmb_mode.currentText()
        eps_0 = self.edt_eps_0.text()
        params['eps_0'] = float(eps_0) if eps_0 else None
        params['alpha'] = float(self.edt_alpha.text())
        params['bound'] = self.cmb_bound.currentText()
        params['eta'] = float(self.edt_eta.text())
        params['dropout_rate'] = float(self.edt_dropout.text())

        if (not isinstance(params['n_realiz'], int)) or (isinstance(params['n_realiz'], int) and not (1 <= params['n_realiz'])): 
            QMessageBox.warning(self, "Invalid number of realization", 
                "Number of realization must be an integer greater than 0.")
            return
        
        if params['norm'] not in ['epsins', 'rmse']:
            QMessageBox.warning(self, "Invalid norm of loss function", 
                "Norm must be 'epsins' or 'rmse'.")
            return

        if params['mode'] not in ['resusb', 'kfold', 'leaveoo']:
            QMessageBox.warning(self, "Invalid validation mode", 
                "Mode must be 'resusb', 'kfold' or 'leaveoo'.")
            return

        if ((not isinstance(params['eps_0'], float) and params['eps_0'] is not None) 
        or ((isinstance(params['eps_0'], float) and params['eps_0'] is not None) 
        and params['eps_0'] < 0)):
            QMessageBox.warning(self, "Invalid threshold epsilon free parameter", 
                "Threshold epsilon value must be None or a real number greater than or equal to 0.")
            return

        if ((not isinstance(params['alpha'], float)) 
        or (isinstance(params['alpha'], float) and not (0 < params['alpha'] < 1))):
            QMessageBox.warning(self, "Invalid alpha", 
                "Alpha must be a real number between 0 and 1 (exclusive).")
            return
        
        if params['bound'] not in ['pacbayes', 'vapnik', 'igp', 'igp_approx']:
            QMessageBox.warning(self, "Invalid bound mode", 
                "Bound mode must be 'pacbayes', 'vapnik', 'igp' or 'igp_approx'.")
            return

        if ((not isinstance(params['eta'], float)) 
        or (isinstance(params['eta'], float) and not (0 < params['eta'] < 1))):
            QMessageBox.warning(self, "Invalid eta free parameter", 
                "Eta must be a real number between 0 and 1 (exclusive).")
            return

        if ((not isinstance(params['dropout_rate'], float)) 
        or (isinstance(params['dropout_rate'], float) 
        and not (0 <= params['dropout_rate'] <= 1))):
            QMessageBox.warning(self, 
                "Invalid dropout rate free parameter of PAC-Bayes",
                "Dropout rate of PAC-Bayes must be a real number between 0 and 1 (inclusive).")
            return

        self.data['params'] = params
        self.data['model'] = None
        self.data['stats'] = None
        self.txt_analysis.setText('')
        self.data['ssa'] = None
        self.run_in_thread(self.compute_analysis, self.show_stats, 
                           title='Computing SAR analysis ...')

    def show_stats(self):
        if self.data['stats'] is None: return
        stats = self.data['stats']
        #if stats['loss'] < stats['thres']:
        #    result = f'There is regression ✅'
        #else:
        #    result = f'There is no regression ❌'
        #text =[result,
        text =[f'Sample size: {self.data['x'].shape[0]}',
              f'Loss: {stats['loss']:.4f}',
              f'Threshold: {stats['thres']:.4f}',
              f'Power: {stats['power']:.4f}']
        self.txt_analysis.setText('\n'.join(text))

    def compute_analysis(self):
        model = sarlib.SAR(**self.data['params'])
        stats = model.fit(self.data['x'], self.data['y'], verbose=False)
        self.data['stats'] = stats
        self.data['model'] = model

    def compute_ssa(self):
        if self.data['ssa'] is not None: return  # Already computed
        if self.data['model'] is None: return    # No model available

        self.data['ssa'] = sarlib.SampleSizeAnalysis(self.data['model'], 
                                                     self.data['x'], 
                                                     self.data['y'], 
                                                     verbose=False)

    def plot_ssa(self, type):
        if self.data['model'] is None:
            QMessageBox.warning(self, "Error", "Please run SAR analysis first.")
            return

        if self.data['ssa'] is None: 
            self.run_in_thread(self.compute_ssa, lambda: self.plot_ssa(type), 
                              title='Computing sample size analysis ...')
        else:
            eval('self.data["ssa"].plot_' + type)()

    def run_in_thread(self, job, after_job, title='Processing ...'):
        def finished_job():
            progress.close()
            after_job()
            self.worker = None

        progress = QProgressDialog(title, None, 0, 0)
        progress.setWindowTitle('Please wait')
        progress.setWindowModality(Qt.ApplicationModal)
        progress.setMinimumDuration(0)
        progress.setCancelButton(None)
        progress.setWindowFlags(Qt.Dialog | Qt.CustomizeWindowHint | Qt.WindowTitleHint)
        progress.show()

        self.worker = Worker(job)
        self.worker.finished.connect(finished_job)
        self.worker.start()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.resize(800, 800)
    w.show()
    sys.exit(app.exec())
