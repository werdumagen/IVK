import os
import tkinter as tk
from gui import FullNMRApp


def main():
    app = FullNMRApp()

    # Splash screen logic (1000 ms duration)
    splash = tk.Toplevel(app)
    splash.overrideredirect(True)

    logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logo.png")
    if not os.path.exists(logo_path):
        logo_path = "logo.png"

    try:
        splash_img = tk.PhotoImage(file=logo_path)
        lbl = tk.Label(splash, image=splash_img, bg="#ffffff")
        lbl.pack()
        w, h = splash_img.width(), splash_img.height()
        splash._img_ref = splash_img
    except Exception:
        lbl = tk.Label(splash, text="1H, 13C, COSY & HSQC NMR Suite\nLoading...",
                       font=("Segoe UI", 16, "bold"), bg="#0f4c81", fg="#ffffff",
                       padx=50, pady=35)
        lbl.pack()
        splash.update_idletasks()
        w, h = splash.winfo_reqwidth(), splash.winfo_reqheight()

    sw = app.winfo_screenwidth()
    sh = app.winfo_screenheight()
    x = max(0, (sw - w) // 2)
    y = max(0, (sh - h) // 2)
    splash.geometry(f"{w}x{h}+{x}+{y}")
    splash.lift()
    splash.attributes('-topmost', True)

    def close_splash():
        splash.destroy()
        app.deiconify()

    app.after(1000, close_splash)
    app.mainloop()


if __name__ == "__main__":
    main()