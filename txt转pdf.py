import tkinter
import tkinter.filedialog
import tkinter.messagebox
import customtkinter
import os
import multiprocessing # Changed from threading
from multiprocessing import Manager, Pool # Use Manager for shared Queue, Pool directly
import queue # Added for queue.Empty exception
# Removed ThreadPoolExecutor import
from fpdf import FPDF
from pathlib import Path
import time
import locale
import math
# Removed datetime and timedelta imports
 
FONT_PATH = ''Alibaba-PuHuiTi-Regular.ttf''
FONT_NAME = 'CustomFont'
# Use CPU count for true parallelism, fallback to 4 if None
MAX_CONCURRENT = os.cpu_count() if os.cpu_count() else 4
BUFFER_SIZE = 16 * 1024 * 1024  # 16MB 缓冲区大小
 
customtkinter.set_appearance_mode("System")
customtkinter.set_default_color_theme("blue")
 
# --- Worker Functions (Top Level for Multiprocessing) ---
def convert_file_worker(txt_path, output_queue):
    """Worker function executed in a separate process."""
    pdf_path = Path(txt_path).with_suffix('.pdf')
    file_basename = os.path.basename(txt_path)
    content = None # Initialize content
 
    try:
        # 尝试读取文件内容和检测编码
        encodings_to_try = ['utf-8', 'gbk', locale.getpreferredencoding(False)]
 
        for enc in encodings_to_try:
            try:
                # Use binary read first for robustness, then decode
                with open(txt_path, 'rb', buffering=BUFFER_SIZE) as f:
                     raw_content = f.read()
                content = raw_content.decode(enc)
                break # 成功读取后退出循环
            except UnicodeDecodeError:
                continue
            except Exception as read_err:
                raise Exception(f"读取时出错 ({txt_path}): {read_err}")
 
        if content is None:
            raise Exception(f"无法使用 {encodings_to_try} 解码文件: {file_basename}")
 
        if not content.strip():
            output_queue.put(("status", txt_path, True, "文件为空，已跳过"))
            return
 
        if not os.path.exists(FONT_PATH):
            raise Exception(f"错误：字体文件 '{FONT_PATH}' 未找到!")
 
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
 
        try:
            pdf.add_font(FONT_NAME, '', FONT_PATH, uni=True)
            pdf.set_font(FONT_NAME, size=10)
        except Exception as font_err:
            raise Exception(f"添加或设置字体时出错 ({file_basename}): {font_err}")
 
        pdf.multi_cell(0, 5, content)
        pdf.output(pdf_path, 'F')
 
        output_queue.put(("status", txt_path, True, f"成功转换为 {os.path.basename(pdf_path)}"))
 
    except Exception as e:
        output_queue.put(("status", txt_path, False, f"处理 {file_basename} 时出错: {e}"))
 
# 将转换进程目标函数移到类外部
def start_conversion_process_target(file_list, queue_ref):
    """Target function for the management process."""
    try:
        # Use context manager for the pool
        with Pool(processes=MAX_CONCURRENT) as pool:
            # Use apply_async to submit tasks without blocking
            for txt_file in file_list:
                pool.apply_async(convert_file_worker, args=(txt_file, queue_ref))
             
            # Close the pool to prevent new tasks
            pool.close()
            # Wait for all worker processes to complete
            pool.join() 
             
        # Signal completion AFTER all workers finished
        queue_ref.put(("done",))
    except Exception as e:
        # Catch errors during pool creation or task submission/joining
        queue_ref.put(("error", f"转换管理进程出错: {e}"))
    # No finally block needed as Manager handles queue lifecycle
 
# --- Main Application Class ---
class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
 
        self.title("TXT to PDF 批量转换器 (MP)") # Simplified title
        self.geometry("700x550") # Reduced height slightly
        self.configure(fg_color=("#f5f5f5", "#2b2b2b"))
 
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(3, weight=1) # Textbox row has weight
 
        main_frame = customtkinter.CTkFrame(self, corner_radius=15, fg_color=("white", "#333333"))
        main_frame.grid(row=0, column=0, padx=30, pady=30, sticky="nsew")
        main_frame.grid_columnconfigure(0, weight=1) # Let column 0 take weight
        main_frame.grid_columnconfigure(1, weight=0) # Column 1 no extra weight
        main_frame.grid_rowconfigure(3, weight=1) # Textbox row has weight
 
        title_label = customtkinter.CTkLabel(
            main_frame,
            text="TXT to PDF 批量转换器",
            font=customtkinter.CTkFont(size=24, weight="bold"),
            text_color=("#1a1a1a", "#ffffff")
        )
        title_label.grid(row=0, column=0, columnspan=2, padx=20, pady=(20, 30))
 
        self.select_folder_button = customtkinter.CTkButton(
            main_frame,
            text="选择 TXT 文件",
            command=self.select_folder,
            height=40,
            corner_radius=8,
            font=customtkinter.CTkFont(size=14),
            fg_color=("#2986cc", "#1f6aa5"),
            hover_color=("#246ea6", "#195785")
        )
        self.select_folder_button.grid(row=1, column=0, padx=(30, 5), pady=(0, 15), sticky="ew")
 
        self.start_button = customtkinter.CTkButton(
            main_frame,
            text="开始转换",
            command=self.start_conversion,
            state="disabled",
            height=40,
            corner_radius=8,
            font=customtkinter.CTkFont(size=14),
            fg_color=("#27ae60", "#219653"),
            hover_color=("#219653", "#1e8449")
        )
        self.start_button.grid(row=1, column=1, padx=(5, 30), pady=(0, 15), sticky="ew")
 
        self.folder_path_label = customtkinter.CTkLabel(
            main_frame,
            text="未选择文件",
            font=customtkinter.CTkFont(size=12),
            text_color=("#666666", "#999999"),
            anchor="w"
        )
        self.folder_path_label.grid(row=2, column=0, columnspan=2, padx=30, pady=(0, 15), sticky="ew")
 
        self.status_textbox = customtkinter.CTkTextbox(
            main_frame,
            state="disabled",
            wrap="word",
            height=200,
            corner_radius=8,
            border_width=1,
            border_color=("#e0e0e0", "#404040"),
            fg_color=("#ffffff", "#2b2b2b")
        )
        self.status_textbox.grid(row=3, column=0, columnspan=2, padx=30, pady=(0, 20), sticky="nsew")
 
        self.progressbar = customtkinter.CTkProgressBar(
            main_frame,
            height=15,
            corner_radius=5,
            fg_color=("#f0f0f0", "#333333"),
            progress_color=("#2986cc", "#1f6aa5")
        )
        self.progressbar.grid(row=4, column=0, columnspan=2, padx=30, pady=(0, 10), sticky="ew")
        self.progressbar.set(0)
 
        # 添加进度百分比标签 (Simplified)
        self.progress_percent_label = customtkinter.CTkLabel(
            main_frame,
            text="0%",
            font=customtkinter.CTkFont(size=12),
            text_color=("#666666", "#999999")
        )
        # Place it below the progress bar, aligned left
        self.progress_percent_label.grid(row=5, column=0, columnspan=2, padx=30, pady=(0, 20), sticky="w") # Changed sticky to 'w'
 
        self.selected_folder = ""
        self.txt_files = []
        self.manager = None # To hold the multiprocessing Manager
        self.conversion_queue = None # Will be created by Manager
        self.conversion_process = None
        self.monitor_queue_id = None
 
        self.start_time = None
        self.processed_files = 0
        self.total_files = 0
        self.failed_files = []
 
    def select_folder(self):
        if self.conversion_process and self.conversion_process.is_alive():
            self.log_status("请等待当前转换完成。")
            return
 
        files = tkinter.filedialog.askopenfilenames(filetypes=[("TXT files", "*.txt")])
        if files:
            self.txt_files = list(files)
            self.selected_folder = os.path.dirname(self.txt_files[0])
            self.folder_path_label.configure(text=f"已选择 {len(self.txt_files)} 个文件")
            self.log_status(f"已选择 {len(self.txt_files)} 个 TXT 文件")
            self.start_button.configure(state="normal")
            self.update_progress(value=0)
            self.processed_files = 0
            self.total_files = len(self.txt_files)
            self.failed_files = []
        else:
            if self.txt_files:
                 self.selected_folder = ""
                 self.txt_files = []
                 self.folder_path_label.configure(text="未选择文件")
                 self.start_button.configure(state="disabled")
                 self.log_status("未选择文件。")
                 self.update_progress(value=0)
 
    def log_status(self, message):
        def _update():
            self.status_textbox.configure(state="normal")
            self.status_textbox.insert("end", f"{message}\n")
            self.status_textbox.configure(state="disabled")
            self.status_textbox.see("end")
        self.after(0, _update)
 
    def update_progress(self, value=None): # Simplified signature
        def _update():
            if value is not None:
                self.progressbar.set(value)
                percent = int(value * 100)
                self.progress_percent_label.configure(text=f"{percent}%")
            # Removed text parameter handling
        self.after(0, _update)
 
    def process_queue(self):
        try:
            while True:
                message = self.conversion_queue.get_nowait()
                msg_type = message[0]
 
                if msg_type == "status":
                    # Unpack only necessary info
                    _, txt_path, success, status_msg = message
                    self.processed_files += 1
                    progress_value = self.processed_files / self.total_files if self.total_files > 0 else 0
                    self.update_progress(value=progress_value)
 
                    log_prefix = "[成功]" if success else "[失败]"
                    self.log_status(f"{log_prefix} {os.path.basename(txt_path)}: {status_msg}")
                    if not success:
                        self.failed_files.append(os.path.basename(txt_path))
 
                elif msg_type == "done":
                    end_time = time.time()
                    duration = end_time - self.start_time if self.start_time else 0
                    self.log_status("-" * 20)
                    self.log_status(f"转换完成！总共处理 {self.processed_files} 个文件，耗时: {duration:.2f} 秒。")
                    if self.failed_files:
                        self.log_status(f"失败 {len(self.failed_files)} 个文件: {', '.join(self.failed_files)}")
                    else:
                        self.log_status("所有文件转换成功！")
                    self.start_button.configure(state="normal")
                    self.select_folder_button.configure(state="normal")
                    self.conversion_process = None
                    self.monitor_queue_id = None
                    self.update_progress(value=1.0)
                    # Stop the manager when done
                    if self.manager:
                        try: # Add try-except for shutdown
                            self.manager.shutdown()
                        except Exception as e:
                            print(f"Error shutting down manager after completion: {e}")
                        self.manager = None
                    return
 
                elif msg_type == "error":
                    self.log_status(f"[严重错误] {message[1]}")
                    self.start_button.configure(state="normal")
                    self.select_folder_button.configure(state="normal")
                    self.conversion_process = None
                    self.monitor_queue_id = None
                    # Stop the manager on error
                    if self.manager:
                        try: # Add try-except for shutdown
                            self.manager.shutdown()
                        except Exception as e:
                            print(f"Error shutting down manager on error: {e}")
                        self.manager = None
                    return
 
        except queue.Empty:
            pass
 
        # Continue monitoring if the process is alive
        if self.conversion_process and self.conversion_process.is_alive():
             self.monitor_queue_id = self.after(100, self.process_queue)
        # Handle unexpected process termination
        elif self.processed_files < self.total_files and self.conversion_process and not self.conversion_process.is_alive():
             self.log_status("[错误] 转换进程意外终止。")
             self.start_button.configure(state="normal")
             self.select_folder_button.configure(state="normal")
             self.conversion_process = None
             self.monitor_queue_id = None
             if self.manager: # Shutdown manager if process died
                 try: # Add try-except for shutdown
                     self.manager.shutdown()
                 except Exception as e:
                     print(f"Error shutting down manager on unexpected process exit: {e}")
                 self.manager = None
        # Process remaining messages after process finished
        elif self.processed_files == self.total_files and self.conversion_queue and not self.conversion_queue.empty():
             # Check if the queue still has items even if process finished
             # This might happen if the 'done' message is still pending
             self.monitor_queue_id = self.after(100, self.process_queue)
 
 
    def start_conversion(self):
        if not self.txt_files:
            self.log_status("请先选择 TXT 文件。")
            return
 
        if self.conversion_process and self.conversion_process.is_alive():
             self.log_status("转换已经在进行中...")
             return
 
        if self.monitor_queue_id:
            self.after_cancel(self.monitor_queue_id)
            self.monitor_queue_id = None
 
        # Clean up previous process and manager if they exist
        if self.conversion_process:
            if self.conversion_process.is_alive():
                self.conversion_process.terminate()
                self.conversion_process.join(timeout=0.5)
            self.conversion_process = None
        if self.manager:
             try:
                 self.manager.shutdown()
             except Exception as e:
                 print(f"Error shutting down old manager: {e}") # Keep error print
             self.manager = None
 
        # Create Manager and managed Queue
        self.manager = Manager()
        self.conversion_queue = self.manager.Queue()
 
        self.start_button.configure(state="disabled")
        self.select_folder_button.configure(state="disabled")
        self.status_textbox.configure(state="normal")
        self.status_textbox.delete("1.0", "end")
        self.status_textbox.configure(state="disabled")
 
        self.update_progress(value=0)
        self.processed_files = 0
        self.total_files = len(self.txt_files)
        self.failed_files = []
 
        self.log_status(f"开始使用 {MAX_CONCURRENT} 个进程转换 {self.total_files} 个文件...")
        self.start_time = time.time()
 
        # Create and start the conversion management process (NOT as daemon)
        self.conversion_process = multiprocessing.Process(
            target=start_conversion_process_target,
            args=(self.txt_files, self.conversion_queue), # Pass manager queue
            daemon=False # IMPORTANT: Set daemon to False
        )
        self.conversion_process.start()
 
        # Start monitoring the queue
        self.monitor_queue_id = self.after(100, self.process_queue)
 
    def on_closing(self):
        if self.conversion_process and self.conversion_process.is_alive():
            if tkinter.messagebox.askyesno("退出", "转换仍在进行中，确定要退出吗？"):
                print("Terminating conversion process...") # Keep termination print
                self.conversion_process.terminate()
                self.conversion_process.join(timeout=0.5)
                if self.manager:
                    try:
                        self.manager.shutdown()
                    except Exception as e:
                        print(f"Error shutting down manager during termination: {e}") # Keep error print
                self.destroy()
            else:
                return
        else:
            if self.manager:
                 try:
                     self.manager.shutdown()
                 except Exception as e:
                     print(f"Error shutting down manager on normal close: {e}") # Keep error print
            self.destroy()
 
if __name__ == "__main__":
    # Required for multiprocessing freeze support on Windows/macOS etc.
    multiprocessing.freeze_support()
 
    if not os.path.exists(FONT_PATH):
        # Use a simple Tk window for the error if CustomTkinter window fails
        root = tkinter.Tk()
        root.withdraw()
        tkinter.messagebox.showerror("字体错误", f"错误：字体文件 '{FONT_PATH}' 未找到! 请确保字体文件路径正确。")
        root.destroy()
        exit()
 
    try:
        app = App()
        app.protocol("WM_DELETE_WINDOW", app.on_closing)
        app.mainloop()
    except Exception as e:
        print(f"Application failed to start: {e}") # Keep error print
        # Fallback error display if app init fails
        root = tkinter.Tk()
        root.withdraw()
        tkinter.messagebox.showerror("启动错误", f"无法启动应用程序: {e}")
        root.destroy()