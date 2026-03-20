# import win32gui
# import win32con
# import tkinter as tk
#
# # 创建主窗口（用tkinter简化UI，实际可替换为其他框架）
# root = tk.Tk()
# root.title("waxeia's note")
# root.geometry("300x280+1400+100")  # 大小和位置
# root.attributes("-alpha", 0.8)  # 半透明
#
# # 获取窗口句柄
# hwnd = root.winfo_id()
#
# # 设置窗口样式：无边框、工具窗口（避免任务栏显示）
# win32gui.SetWindowLong(hwnd, win32con.GWL_STYLE,
#                        win32con.WS_POPUP | win32con.WS_EX_TOOLWINDOW)
#
# # 关键：设置窗口层级为“桌面之上，普通应用之下”
# # HWND_BOTTOM 会置于所有窗口下方，但需配合桌面窗口的关系调整
# # 更精准的方式是找到桌面窗口（Progman），将编辑区置于其上方
# progman = win32gui.FindWindow("Progman", "Program Manager")
# win32gui.SetWindowPos(hwnd, progman, 0, 0, 0, 0,
#                       win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_NOACTIVATE)
#
# # 添加编辑控件（如文本框）
# text = tk.Text(root, font=("SimHei", 12), bg="#FFEEEE")
# text.pack(fill="both", expand=True)
#
# root.mainloop()


import win32gui
import win32con
import tkinter as tk
import os
import sys

# --- 辅助函数：获取应用数据存储的正确目录 ---
def get_app_data_path():
    """
    获取应用数据存储的目录路径，兼容开发环境和打包后的exe。
    返回一个绝对路径。
    """
    if getattr(sys, 'frozen', False):
        # 如果是打包后的exe，sys.frozen 会被设为 True
        # 获取exe文件所在的目录
        base_path = os.path.dirname(sys.executable)
    else:
        # 如果是开发环境（直接运行.py文件）
        # 获取当前脚本文件所在的目录
        base_path = os.path.dirname(os.path.abspath(__file__))
    return base_path

# --- 数据持久化设置 ---
# 获取应用数据目录，并构建笔记文件的完整路径
APP_DATA_DIR = get_app_data_path()
NOTE_FILE = os.path.join(APP_DATA_DIR, "note_content.txt")

# --- 程序关闭时的保存函数 ---
def save_on_exit():
    """在窗口关闭前，将文本内容保存到文件"""
    try:
        content = text.get("1.0", "end-1c")
        # 直接使用全局定义的 NOTE_FILE 路径
        with open(NOTE_FILE, "w", encoding="utf-8") as f:
            f.write(content)
    except Exception as e:
        # 在开发时，如果出错可以打印到控制台查看
        # 打包后，这个打印信息不可见，但通常不会有问题
        print(f"保存文件时出错: {e}")
    finally:
        # 无论保存是否成功，最后都要销毁窗口以退出程序
        root.destroy()

# --- 主程序开始 ---

# 创建主窗口
root = tk.Tk()
root.title("waxeia's note")
root.geometry("300x280+1400+100")
root.attributes("-alpha", 0.8)

# 获取窗口句柄
hwnd = root.winfo_id()

# 设置窗口样式
win32gui.SetWindowLong(hwnd, win32con.GWL_STYLE,
                       win32con.WS_POPUP | win32con.WS_EX_TOOLWINDOW)

# 设置窗口层级
progman = win32gui.FindWindow("Progman", "Program Manager")
win32gui.SetWindowPos(hwnd, progman, 0, 0, 0, 0,
                      win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_NOACTIVATE)

# 添加编辑控件
text = tk.Text(root, font=("SimHei", 12), bg="#FFEEEE")
text.pack(fill="both", expand=True)

# --- 程序启动时的加载逻辑 ---
# 直接使用全局定义的 NOTE_FILE 路径
if os.path.exists(NOTE_FILE):
    try:
        with open(NOTE_FILE, "r", encoding="utf-8") as f:
            saved_content = f.read()
            text.insert("1.0", saved_content)
    except Exception as e:
        print(f"读取文件时出错: {e}")

# 注册窗口关闭事件
root.protocol("WM_DELETE_WINDOW", save_on_exit)

root.mainloop()