import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel
from PyQt5.QtGui import QFont, QColor, QPalette
from PyQt5.QtCore import Qt

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("My PyQt5 App")
        self.setGeometry(300, 300, 400, 300)

        # 设置整体布局
        layout = QVBoxLayout()

        # 创建一个标题标签
        title_label = QLabel("Welcome to My App")
        title_label.setFont(QFont("Arial", 24, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        # 创建按钮
        button = QPushButton("Click Me")
        button.setFont(QFont("Arial", 14))
        button.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px;")
        button.clicked.connect(self.on_button_click)
        layout.addWidget(button)

        # 设置中心窗口的小部件
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # 设置窗口背景颜色
        self.setStyleSheet("background-color: #f0f0f0;")

    def on_button_click(self):
        # 当按钮被点击时执行的操作
        self.setWindowTitle("Button Clicked!")

# 主程序入口
if __name__ == "__main__":
    app = QApplication(sys.argv)

    # 创建主窗口
    main_window = MainWindow()
    main_window.show()

    # 启动应用程序主循环
    sys.exit(app.exec_())
