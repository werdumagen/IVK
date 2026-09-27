# main.py
import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QTabWidget
from tab_setup import VisualizerTab
from tab_match import MatcherTab

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NMR Topology & Anchor Matching Engine")
        self.resize(1700, 1000)
        self._init_ui()

    def _init_ui(self):
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Создаем экземпляры вкладок
        self.tab_setup = VisualizerTab()
        self.tab_match = MatcherTab()

        self.tabs.addTab(self.tab_setup, "1. Подготовка и Графы")
        self.tabs.addTab(self.tab_match, "2. Строгое Сопоставление (Якоря)")

        # Подключаем сигнал готовности данных из первой вкладки во вторую
        self.tab_setup.requestMatching.connect(self.switch_to_matching)

    def switch_to_matching(self, graphs_data: dict, mol):
        # 1. Передаем объекты напрямую в память вкладки сопоставления
        self.tab_match.load_data_and_match(graphs_data, mol)
        # 2. Автоматически переключаем активную вкладку
        self.tabs.setCurrentIndex(1)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())