import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import hashlib
import random
import requests
import json
import sys
from urllib.parse import quote
import platform

# 根据操作系统选择合适的鼠标位置获取方法
if platform.system() == 'Windows':
    import ctypes
    from ctypes import wintypes

    # Windows API 获取鼠标位置
    user32 = ctypes.windll.user32


    class POINT(ctypes.Structure):
        _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


    def get_mouse_position():
        """获取鼠标位置（Windows）"""
        pt = POINT()
        user32.GetCursorPos(ctypes.byref(pt))
        return pt.x, pt.y
else:
    # 其他系统使用原始方法
    def get_mouse_position():
        """获取鼠标位置（其他系统）"""
        try:
            import pyautogui
            return pyautogui.position()
        except:
            return 0, 0


class SmartBaiduTranslator:
    def __init__(self):
        # 百度翻译API配置（请替换为您自己的APP ID和密钥）
        self.appid = '20251030002486993'  # 请替换为您的百度翻译APP ID
        self.secret_key = 'aFuJbmMN0RwN55D0XoXE'  # 请替换为您的百度翻译密钥
        self.api_url = 'https://fanyi-api.baidu.com/api/trans/vip/translate'

        # 窗口尺寸和位置参数
        self.window_width = 320
        self.window_height = 450
        self.visible_width = 320  # 完全显示时的宽度
        self.hidden_width = 15  # 隐藏时显示的宽度
        self.position = "right"  # 可以是 "left", "right"

        # 动画相关变量
        self.is_visible = True
        self.animation_speed = 15  # 动画速度（毫秒）
        self.hide_delay = 1500  # 鼠标离开后延迟隐藏时间（毫秒）
        self.hide_timer = None

        self.window_center_y = 0

        # 鼠标监控相关
        self.last_mouse_in_area = False
        self.last_check_time = 0

        # 创建主窗口
        self.root = tk.Tk()
        self.setup_window()
        self.setup_ui()

        # 设置初始位置
        self.set_window_position()

        # 启动鼠标监控
        self.start_mouse_monitor()

        # 启动主循环
        self.root.mainloop()

    def setup_window(self):
        """设置窗口属性"""
        self.root.title("智能翻译")
        self.root.overrideredirect(True)  # 无边框窗口
        self.root.attributes('-topmost', True)  # 始终置顶
        self.root.attributes('-alpha', 0.95)  # 设置透明度

        # 设置背景色
        self.root.configure(bg='#1e1e1e')

        # 设置窗口样式
        self.root.resizable(False, False)

    def setup_ui(self):
        """设置用户界面"""
        # 主容器
        main_container = tk.Frame(self.root, bg='#1e1e1e')
        main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 标题栏（用于拖动）
        title_bar = tk.Frame(main_container, bg='#2d2d2d', height=35)
        title_bar.pack(fill=tk.X, pady=(0, 10))
        title_bar.pack_propagate(False)

        # 标题文字
        title_label = tk.Label(
            title_bar,
            text="🌐 智能翻译",
            bg='#2d2d2d',
            fg='white',
            font=('Microsoft YaHei', 11, 'bold')
        )
        title_label.pack(side=tk.LEFT, padx=10, pady=8)

        # 状态指示器
        self.status_label = tk.Label(
            title_bar,
            text="●",
            bg='#2d2d2d',
            fg='#4CAF50',
            font=('Arial', 10)
        )
        self.status_label.pack(side=tk.LEFT, padx=5, pady=8)

        # 最小化按钮
        minimize_btn = tk.Button(
            title_bar,
            text="─",
            bg='#2d2d2d',
            fg='white',
            font=('Arial', 12),
            bd=0,
            activebackground='#3d3d3d',
            activeforeground='white',
            command=self.minimize_window,
            width=3
        )
        minimize_btn.pack(side=tk.RIGHT, padx=2, pady=5)

        # 关闭按钮
        close_btn = tk.Button(
            title_bar,
            text="✕",
            bg='#2d2d2d',
            fg='white',
            font=('Arial', 12),
            bd=0,
            activebackground='#e74c3c',
            activeforeground='white',
            command=self.close_app,
            width=3
        )
        close_btn.pack(side=tk.RIGHT, padx=2, pady=5)

        # 绑定拖动事件
        title_bar.bind('<Button-1>', self.start_drag)
        title_bar.bind('<B1-Motion>', self.on_drag)
        title_label.bind('<Button-1>', self.start_drag)
        title_label.bind('<B1-Motion>', self.on_drag)

        # 语言显示区域
        lang_frame = tk.Frame(main_container, bg='#1e1e1e')
        lang_frame.pack(fill=tk.X, pady=(0, 10))

        self.lang_display = tk.Label(
            lang_frame,
            text="自动检测 → 智能翻译",
            bg='#2d2d2d',
            fg='#FFA500',
            font=('Microsoft YaHei', 9),
            padx=10,
            pady=5
        )
        self.lang_display.pack(fill=tk.X)

        # 输入区域
        input_container = tk.Frame(main_container, bg='#1e1e1e')
        input_container.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        input_label = tk.Label(
            input_container,
            text="输入文本：",
            bg='#1e1e1e',
            fg='#cccccc',
            font=('Microsoft YaHei', 9)
        )
        input_label.pack(anchor=tk.W)

        # 输入文本框
        self.input_text = tk.Text(
            input_container,
            height=7,
            bg='#2d2d2d',
            fg='white',
            font=('Microsoft YaHei', 10),
            insertbackground='white',
            relief=tk.FLAT,
            wrap=tk.WORD,
            bd=0,
            padx=10,
            pady=5
        )
        self.input_text.pack(fill=tk.BOTH, expand=True, pady=(5, 0))

        # 绑定输入事件
        self.input_text.bind('<KeyRelease>', self.on_input_change)
        self.input_text.bind('<Control-Return>', lambda e: self.translate())
        self.input_text.bind('<Control-Enter>', lambda e: self.translate())

        # 翻译按钮
        self.translate_btn = tk.Button(
            main_container,
            text="翻译 (Ctrl+Enter)",
            bg='#1890ff',
            fg='white',
            font=('Microsoft YaHei', 10, 'bold'),
            bd=0,
            activebackground='#40a9ff',
            command=self.translate,
            height=2,
            cursor='hand2'
        )
        self.translate_btn.pack(fill=tk.X, pady=(0, 10))

        # 输出区域
        output_container = tk.Frame(main_container, bg='#1e1e1e')
        output_container.pack(fill=tk.BOTH, expand=True)

        output_label = tk.Label(
            output_container,
            text="翻译结果：",
            bg='#1e1e1e',
            fg='#cccccc',
            font=('Microsoft YaHei', 9)
        )
        output_label.pack(anchor=tk.W)

        # 输出文本框
        self.output_text = tk.Text(
            output_container,
            height=7,
            bg='#2d2d2d',
            fg='white',
            font=('Microsoft YaHei', 10),
            relief=tk.FLAT,
            wrap=tk.WORD,
            state=tk.DISABLED,
            bd=0,
            padx=10,
            pady=5
        )
        self.output_text.pack(fill=tk.BOTH, expand=True, pady=(5, 0))

        # 按钮框架
        button_frame = tk.Frame(output_container, bg='#1e1e1e')
        button_frame.pack(fill=tk.X, pady=(5, 0))

        # 复制按钮
        copy_btn = tk.Button(
            button_frame,
            text="📋 复制",
            bg='#3d3d3d',
            fg='white',
            font=('Microsoft YaHei', 9),
            bd=0,
            activebackground='#4d4d4d',
            command=self.copy_result,
            cursor='hand2'
        )
        copy_btn.pack(side=tk.RIGHT)

        # 清空按钮
        clear_btn = tk.Button(
            button_frame,
            text="🗑 清空",
            bg='#3d3d3d',
            fg='white',
            font=('Microsoft YaHei', 9),
            bd=0,
            activebackground='#4d4d4d',
            command=self.clear_text,
            cursor='hand2'
        )
        clear_btn.pack(side=tk.RIGHT, padx=(0, 5))

        # 添加一个拖动手柄（在隐藏时可见）
        self.drag_handle = tk.Label(
            self.root,
            text="◀",
            bg='#4CAF50',
            fg='white',
            font=('Arial', 12),
            cursor='hand2'
        )

        # 根据窗口位置设置手柄的初始位置
        if self.position == "right":
            handle_x = self.visible_width - 15
            handle_text = "▶"
        else:
            handle_x = 0
            handle_text = "◀"

        self.drag_handle.place(x=handle_x, y=self.window_height // 2 - 10, width=15, height=20)
        self.drag_handle.config(text=handle_text)

        # 拖动手柄的事件绑定
        self.drag_handle.bind('<Button-1>', lambda e: self.show_window())
        self.drag_handle.bind('<Enter>', lambda e: self.show_window())

        # 窗口事件绑定
        self.root.bind('<Enter>', lambda e: self.on_window_enter(e))
        self.root.bind('<Leave>', lambda e: self.on_window_leave(e))

    def on_window_enter(self, event):
        # 确保手柄可见
        self.drag_handle.lift()
        self.show_window()

    def on_window_leave(self, event):
        # 只在鼠标真正离开窗口区域时才处理
        x, y = get_mouse_position()
        x1, y1 = self.root.winfo_rootx(), self.root.winfo_rooty()
        x2, y2 = x1 + self.root.winfo_width(), y1 + self.root.winfo_height()

        # 鼠标完全离开窗口时才隐藏主窗口，但保留手柄
        if not (x1 <= x <= x2 and y1 <= y <= y2):
            # 隐藏主窗口内容但保留手柄
            pass

    def check_if_leave(self):
        # 获取鼠标位置
        x, y = get_mouse_position()
        # 获取窗口位置和大小
        x1, y1 = self.root.winfo_rootx(), self.root.winfo_rooty()
        x2, y2 = x1 + self.root.winfo_width(), y1 + self.root.winfo_height()

        # 如果鼠标完全窗口范围内才隐藏
        if not (x1 <= x <= x2 and y1 <= y <= y2):
            self.hide_window()

    def start_mouse_monitor(self):
        """启动鼠标监控"""
        self.monitor_mouse()

    def monitor_mouse(self):
        """监控鼠标位置"""
        try:
            # 使用更可靠的方法获取鼠标位置
            mouse_x, mouse_y = get_mouse_position()

            # 获取屏幕尺寸
            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()

            # 获取窗口位置
            try:
                win_x = self.root.winfo_x()
                win_y = self.root.winfo_y()
                win_right = win_x + self.root.winfo_width()
                win_bottom = win_y + self.root.winfo_height()
            except:
                # 如果获取窗口位置失败，使用缓存的位置
                win_x = getattr(self, '_cached_win_x', screen_width - self.visible_width)
                win_y = getattr(self, '_cached_win_y', (screen_height - self.window_height) // 2)
                win_right = win_x + self.root.winfo_width()
                win_bottom = win_y + self.window_height

            # 缓存窗口位置
            self._cached_win_x = win_x
            self._cached_win_y = win_y

            mouse_in_area = False

            if self.position == "right":
                # 右侧边缘检测区域
                trigger_x_start = screen_width - 5
                trigger_x_end = screen_width

                # 当窗口隐藏时，使用更大的Y轴检测范围
                if not self.is_visible:
                    # 隐藏时：在窗口中心上下各200像素的范围内都可以触发
                    trigger_y_start = max(0, self.window_center_y - 200)
                    trigger_y_end = min(screen_height, self.window_center_y + 200)

                    if mouse_x >= trigger_x_start and mouse_x <= trigger_x_end:
                        if mouse_y >= trigger_y_start and mouse_y <= trigger_y_end:
                            mouse_in_area = True
                else:
                    # 显示时：使用窗口实际位置
                    # 扩大检测区域到窗口右侧60像素
                    if mouse_x >= win_right - 60 and mouse_x <= screen_width:
                        if mouse_y >= win_y and mouse_y <= win_bottom:
                            mouse_in_area = True

            else:  # left
                # 左侧边缘检测区域
                trigger_x_start = 0
                trigger_x_end = 5

                # 当窗口隐藏时，使用更大的Y轴检测范围
                if not self.is_visible:
                    # 隐藏时：在窗口中心上下各200像素的范围内都可以触发
                    trigger_y_start = max(0, self.window_center_y - 200)
                    trigger_y_end = min(screen_height, self.window_center_y + 200)

                    if mouse_x >= trigger_x_start and mouse_x <= trigger_x_end:
                        if mouse_y >= trigger_y_start and mouse_y <= trigger_y_end:
                            mouse_in_area = True
                else:
                    # 显示时：使用窗口实际位置
                    # 扩大检测区域到窗口左侧60像素
                    if mouse_x >= 0 and mouse_x <= 60:
                        if mouse_y >= win_y and mouse_y <= win_bottom:
                            mouse_in_area = True

            # 根据鼠标位置执行相应操作
            if mouse_in_area:
                # 鼠标进入激活区域
                if not self.is_visible:
                    self.show_window()
                if self.hide_timer:
                    self.root.after_cancel(self.hide_timer)
                    self.hide_timer = None
                self.last_mouse_in_area = True
            else:
                # 鼠标离开激活区域
                if self.last_mouse_in_area and self.is_visible:
                    if self.hide_timer:
                        self.root.after_cancel(self.hide_timer)
                    self.hide_timer = self.root.after(self.hide_delay, self.hide_window)
                self.last_mouse_in_area = False

        except Exception as e:
            # 出错时也要继续监控
            print(f"Mouse monitor error: {e}")
            pass

        # 每100毫秒检查一次
        self.root.after(100, self.monitor_mouse)

    def set_window_position(self):
        """设置窗口位置"""
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        if self.position == "right":
            x = screen_width - self.visible_width
            y = (screen_height - self.window_height) // 2
        else:  # left
            x = 0
            y = (screen_height - self.window_height) // 2

        self.window_center_y = y + self.window_height // 2

        self.root.geometry(f"{self.visible_width}x{self.window_height}+{x}+{y}")

    # 修复show_window方法，添加滑入动画
    def show_window(self):
        """显示完整窗口（带滑入动画）"""
        if self.is_visible:
            return

        self.is_visible = True

        # 停止任何正在进行的隐藏计时器
        if self.hide_timer:
            self.root.after_cancel(self.hide_timer)
            self.hide_timer = None

        # 根据位置执行相应的滑入动画
        if self.position == "right":
            self.slide_in_from_right()
        else:
            self.slide_in_from_left()

    def hide_window(self):
        """隐藏窗口动画"""
        if not self.is_visible:
            return

        self.is_visible = False

        if self.position == "right":
            self.slide_out_to_right()
        else:
            self.slide_out_to_left()

    def slide_in_from_right(self):
        """从右侧滑入"""
        screen_width = self.root.winfo_screenwidth()
        target_x = screen_width - self.visible_width
        current_x = self.root.winfo_x()

        if current_x > target_x:  # 修改条件：当前x大于目标x时继续移动
            new_x = max(current_x - 25, target_x)  # 修改：减去25而不是加上
            self.root.geometry(f"+{new_x}+{self.root.winfo_y()}")
            self.root.after(self.animation_speed, self.slide_in_from_right)
        else:
            # 动画完成后隐藏手柄
            self.drag_handle.place_forget()

    def slide_out_to_right(self):
        """向右侧滑出"""
        screen_width = self.root.winfo_screenwidth()
        target_x = screen_width - self.hidden_width
        current_x = self.root.winfo_x()

        if current_x < target_x:  # 修改条件：当前x小于目标x时继续移动
            new_x = min(current_x + 25, target_x)  # 修改：加上25而不是减去
            self.root.geometry(f"+{new_x}+{self.root.winfo_y()}")
            self.root.after(self.animation_speed, self.slide_out_to_right)
        else:
            # 显示拖动手柄 - 修正位置计算
            # 手柄应该显示在屏幕的最右侧，而不是窗口的左侧
            handle_x = self.visible_width - 15  # 相对于窗口的位置
            self.drag_handle.place(x=handle_x, y=self.window_height // 2 - 10, width=15, height=20)
            self.drag_handle.config(text="▶")  # 修改箭头方向

    def slide_in_from_left(self):
        """从左侧滑入"""
        target_x = 0
        current_x = self.root.winfo_x()

        if current_x < target_x:  # 修改条件：当前x小于目标x时继续移动
            new_x = min(current_x + 25, target_x)  # 修改：加上25而不是减去
            self.root.geometry(f"+{new_x}+{self.root.winfo_y()}")
            self.root.after(self.animation_speed, self.slide_in_from_left)
        else:
            # 动画完成后隐藏手柄
            self.drag_handle.place_forget()

    def slide_out_to_left(self):
        """向左侧滑出"""
        target_x = -(self.visible_width - self.hidden_width)
        current_x = self.root.winfo_x()

        if current_x > target_x:  # 修改条件：当前x大于目标x时继续移动
            new_x = max(current_x - 25, target_x)  # 修改：减去25而不是加上
            self.root.geometry(f"+{new_x}+{self.root.winfo_y()}")
            self.root.after(self.animation_speed, self.slide_out_to_left)
        else:
            # 显示拖动手柄 - 修正位置计算
            # 手柄应该显示在屏幕的最左侧，而不是窗口的右侧
            handle_x = 0  # 相对于窗口的位置
            self.drag_handle.place(x=handle_x, y=self.window_height // 2 - 10, width=15, height=20)
            self.drag_handle.config(text="◀")  # 修改箭头方向

    def detect_language(self, text):
        """检测输入文本的语言"""
        if not text:
            return "auto", "auto"

        # 简单的语言检测逻辑
        chinese_chars = 0
        english_chars = 0

        for char in text:
            if '\u4e00' <= char <= '\u9fff':
                chinese_chars += 1
            elif char.isalpha() and char <= 'z' and char >= 'a':
                english_chars += 1
            elif char.isalpha() and char <= 'Z' and char >= 'A':
                english_chars += 1

        if chinese_chars > english_chars:
            return "zh", "en"  # 中文输入，翻译成英文
        elif english_chars > 0:
            return "en", "zh"  # 英文输入，翻译成中文
        else:
            return "auto", "auto"

    def on_input_change(self, event):
        """输入改变时更新语言显示"""
        text = self.input_text.get("1.0", tk.END).strip()
        if text:
            source, target = self.detect_language(text)
            if source == "zh":
                self.lang_display.config(text="中文 → 英语", fg='#4CAF50')
            elif source == "en":
                self.lang_display.config(text="英语 → 中文", fg='#1890ff')
            else:
                self.lang_display.config(text="自动检测 → 智能翻译", fg='#FFA500')
        else:
            self.lang_display.config(text="自动检测 → 智能翻译", fg='#FFA500')

    def translate(self):
        """执行翻译"""
        text = self.input_text.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("提示", "请输入要翻译的文本")
            return

        # 检查API配置
        if self.appid == '您的APPID' or self.secret_key == '您的密钥':
            messagebox.showerror("错误", "请先配置百度翻译API的APP ID和密钥！\n\n"
                                         "1. 访问 https://fanyi-api.baidu.com/\n"
                                         "2. 注册账号并获取APP ID和密钥\n"
                                         "3. 修改代码中的appid和secret_key")
            return

        # 智能检测语言
        source, target = self.detect_language(text)

        # 更新状态指示器
        self.status_label.config(fg='#FFA500')
        self.translate_btn.config(text="翻译中...", bg='#FFA500')

        # 在新线程中执行翻译
        threading.Thread(target=self._translate_thread, args=(text, source, target), daemon=True).start()

    def _translate_thread(self, text, source, target):
        """翻译线程"""
        try:
            # 生成salt
            salt = str(random.randint(32768, 65536))

            # 生成sign
            sign_str = self.appid + text + salt + self.secret_key
            sign = hashlib.md5(sign_str.encode('utf-8')).hexdigest()

            # 构建请求参数
            params = {
                'q': text,
                'from': source,
                'to': target,
                'appid': self.appid,
                'salt': salt,
                'sign': sign
            }

            # 发送请求
            response = requests.get(self.api_url, params=params, timeout=10)
            result = response.json()

            # 处理响应
            if 'error_code' in result:
                error_messages = {
                    '52001': '请求超时',
                    '52002': '系统错误',
                    '52003': '未授权用户',
                    '54000': '必填参数为空',
                    '54001': '签名错误',
                    '54003': '访问频率受限',
                    '54004': '账户余额不足',
                    '54005': '长query请求频繁',
                    '58000': '客户端IP非法',
                    '58001': '译文语言方向不支持',
                    '58002': '服务当前已关闭',
                    '90107': '认证未通过或未生效'
                }
                error_msg = error_messages.get(str(result['error_code']), '未知错误')
                self.root.after(0, lambda: self.show_error(error_msg))
            else:
                # 获取翻译结果
                if 'trans_result' in result:
                    translated_text = '\n'.join([item['dst'] for item in result['trans_result']])
                    self.root.after(0, self.update_result, translated_text)

                    # 获取实际检测到的语言
                    if 'from' in result:
                        detected_lang = result['from']
                        if detected_lang == 'zh':
                            self.root.after(0, lambda: self.lang_display.config(text="中文 → 英语", fg='#4CAF50'))
                        elif detected_lang == 'en':
                            self.root.after(0, lambda: self.lang_display.config(text="英语 → 中文", fg='#1890ff'))
                else:
                    self.root.after(0, lambda: self.show_error("翻译结果格式错误"))

        except requests.exceptions.Timeout:
            self.root.after(0, lambda: self.show_error("请求超时，请检查网络连接"))
        except requests.exceptions.ConnectionError:
            self.root.after(0, lambda: self.show_error("网络连接错误，请检查网络"))
        except Exception as e:
            self.root.after(0, lambda: self.show_error(f"翻译失败：{str(e)}"))

    def show_error(self, message):
        """显示错误信息"""
        self.status_label.config(fg='#e74c3c')
        self.translate_btn.config(text="翻译 (Ctrl+Enter)", bg='#e74c3c')
        messagebox.showerror("错误", message)
        # 恢复按钮状态
        self.root.after(2000, lambda: self.translate_btn.config(text="翻译 (Ctrl+Enter)", bg='#1890ff'))
        self.root.after(2000, lambda: self.status_label.config(fg='#4CAF50'))

    def update_result(self, result):
        """更新翻译结果"""
        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert("1.0", result)
        self.output_text.config(state=tk.DISABLED)

        # 恢复按钮状态
        self.translate_btn.config(text="翻译 (Ctrl+Enter)", bg='#1890ff')
        self.status_label.config(fg='#4CAF50')

        # 显示成功提示
        self.show_tooltip("翻译完成")

    def copy_result(self):
        """复制翻译结果"""
        result = self.output_text.get("1.0", tk.END).strip()
        if result:
            self.root.clipboard_clear()
            self.root.clipboard_append(result)
            self.show_tooltip("已复制到剪贴板")

    def clear_text(self):
        """清空文本"""
        self.input_text.delete("1.0", tk.END)
        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete("1.0", tk.END)
        self.output_text.config(state=tk.DISABLED)
        self.lang_display.config(text="自动检测 → 智能翻译", fg='#FFA500')

    def show_tooltip(self, message):
        """显示提示信息"""
        tooltip = tk.Toplevel(self.root)
        tooltip.overrideredirect(True)
        tooltip.attributes('-topmost', True)

        label = tk.Label(
            tooltip,
            text=message,
            bg='#4CAF50',
            fg='white',
            font=('Microsoft YaHei', 9),
            padx=10,
            pady=5
        )
        label.pack()

        # 计算位置
        x = self.root.winfo_x() + (self.window_width - label.winfo_reqwidth()) // 2
        y = self.root.winfo_y() + self.window_height // 2
        tooltip.geometry(f"+{x}+{y}")

        # 2秒后自动关闭
        tooltip.after(2000, tooltip.destroy)

    def start_drag(self, event):
        """开始拖动窗口"""
        self.drag_start_x = event.x
        self.drag_start_y = event.y

    def on_drag(self, event):
        """拖动窗口"""
        x = self.root.winfo_x() + (event.x - self.drag_start_x)
        y = self.root.winfo_y() + (event.y - self.drag_start_y)
        self.root.geometry(f"+{x}+{y}")

        self.window_center_y = y + self.window_height // 2

    def minimize_window(self):
        """最小化窗口"""
        self.hide_window()

    def close_app(self):
        """关闭应用"""
        if messagebox.askyesno("确认", "确定要退出翻译器吗？"):
            self.root.quit()
            self.root.destroy()


if __name__ == "__main__":
    app = SmartBaiduTranslator()