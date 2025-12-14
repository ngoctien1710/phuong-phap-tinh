# Ờm....
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from sympy.calculus.util import continuous_domain
from prettytable import PrettyTable
import os

class NewtonSolver:
    """
    Class giải phương trình f(x) = 0 bằng phương pháp Newton.
    """
    def __init__(self):
        self.x = sp.symbols('x')

    # --- CÁC PHƯƠNG THỨC TÍNH TOÁN " ---

    def _cackhoangphanly(self, f_np, df_np, df_symbolic, xa, xb, step):
        """
        Tìm các khoảng phân ly nghiệm.
        """
        intervals = []
        x1 = float(xa)
        try: # Cố gắng tính giá trị ban đầu
            y1 = f_np(x1)
            dy1 = df_np(x1)
        except (ValueError, ZeroDivisionError, OverflowError):
            y1, dy1 = np.nan, np.nan

        while x1 < xb:
            try:
                # Điều chỉnh bước nhảy linh hoạt dựa trên độ dốc
                if np.isnan(dy1): slope = 1.0
                else: slope = abs(dy1)
                
                dynamic_step = max(0.005, step / (1 + slope))
                
                if slope < 0.01: # Tăng tốc ở vùng gần cực trị
                    dynamic_step = 0.01
                
                x2 = x1 + dynamic_step
                if x2 > xb: x2 = float(xb)
                
                y2 = f_np(x2)
                dy2 = df_np(x2)

                # Bỏ qua nếu gặp giá trị không hợp lệ (NaN)
                if np.isnan(y1) or np.isnan(y2) or np.isnan(dy1) or np.isnan(dy2):
                    x1, y1, dy1 = x2, y2, dy2
                    continue
                
                # 1. Kiểm tra nghiệm thông thường (f(x) đổi dấu)
                if y1 * y2 < 0 or (y2 == 0 and y1 != 0):
                    intervals.append((x1, x2))

                # 2. Săn lùng nghiệm kép khi đạo hàm đổi dấu (dấu hiệu của điểm cực trị)
                elif dy1 * dy2 < 0:
                    # Dùng sp.solve() để tìm chính xác nghiệm của f'(x) = 0, ăn gian tí :D
                    try:
                        potential_extremums = sp.solve(df_symbolic, self.x)
                        # Lọc các nghiệm thực nằm trong khoảng quét nhỏ [x1, x2]
                        real_extremums_in_interval = [
                            float(s) for s in potential_extremums 
                            if s.is_real and x1 <= float(s) <= x2
                        ]
                        
                        for x_cuc_tri in real_extremums_in_interval:
                            y_cuc_tri = f_np(x_cuc_tri)
                            # Nếu tại điểm cực trị, f(x) rất gần 0 -> đây là nghiệm
                            if not np.isnan(y_cuc_tri) and abs(y_cuc_tri) < 0.0001:
                                # Tạo một khoảng nhỏ nhân tạo quanh nghiệm
                                intervals.append((x_cuc_tri - 0.01, x_cuc_tri + 0.01))
                    except (NotImplementedError, TypeError):
                        # Bỏ qua nếu sp.solve không giải được phương trình đạo hàm
                        pass
                
                if x1 == x2: break
                x1, y1, dy1 = x2, y2, dy2

            except (ValueError, ZeroDivisionError, OverflowError):
                # Xử lý lỗi tính toán bằng cách bước một bước nhỏ
                x1 += 0.005
                try: 
                    y1 = f_np(x1)
                    dy1 = df_np(x1)
                except: y1, dy1 = np.nan, np.nan
                continue
        
        # Xử lý gộp các khoảng bị trùng lặp hoặc giao nhau
        if not intervals: return None
        intervals.sort()
        merged = [intervals[0]]
        for current_start, current_end in intervals[1:]:
            last_start, last_end = merged[-1]
            if current_start < last_end: 
                merged[-1] = (last_start, max(last_end, current_end))
            else: 
                merged.append((current_start, current_end))
        return merged

    def _tim_x0(self, f_np, d2f_np, interval):
        a, b = interval
        try:
            # Ưu tiên chọn điểm thỏa mãn điều kiện Fourier
            if f_np(a) * d2f_np(a) > 0: return a
            if f_np(b) * d2f_np(b) > 0: return b
            # Nếu không, chọn điểm có |f(x)| nhỏ hơn
            if abs(f_np(a)) < abs(f_np(b)): return a
            else: return b
        except (ValueError, ZeroDivisionError):
            # Lựa chọn dự phòng
            return (a + b) / 2

    def _pp_newton(self, f_np, df_np, x0, tol):
        steps = [x0]
        table = PrettyTable(["n", "x", "Sai số"])
        table.add_row([0, f"{x0:.7f}", " "])
        x_current = x0
        for i in range(1, 101):
            try:
                f0, df0 = f_np(x_current), df_np(x_current)
                # Dừng nếu đạo hàm quá gần 0 (tiệm cận ngang hoặc nghiệm kép)
                if abs(df0) < 1e-12: return x_current, table, steps
                x_next = x_current - f0 / df0
                error = abs(x_next - x_current)
                steps.append(x_next)
                table.add_row([i, f"{x_next:.7f}", f"{error:.7f}"])
                if error < tol:
                    return x_next, table, steps
                x_current = x_next
            except (ValueError, ZeroDivisionError):
                return None, table, steps # Dừng nếu có lỗi tính toán
        return x_current, table, steps # Trả về kết quả cuối cùng sau 100 lần lặp

    def _generate_and_save_graph(self, f_np, f_expr, interval, steps, solution_index):
        a, b = interval
        # Mở rộng khoảng vẽ đồ thị một chút để nhìn rõ hơn
        plot_a, plot_b = a - 0.2*(b-a), b + 0.2*(b-a)
        x_vals = np.linspace(plot_a, plot_b, 400)
        y_vals = f_np(x_vals)
        
        plt.figure(figsize=(8, 6))
        plt.plot(x_vals, y_vals, label=f'f(x) = {f_expr}', color="blue")
        plt.axhline(0, color='black', linewidth=0.5)
        
        if steps:
            valid_steps = []
            for s in steps:
                try:
                    y_s = f_np(s)
                    if not np.isnan(y_s):
                        valid_steps.append((s, y_s))
                except:
                    continue # Bỏ qua nếu không tính được f(s)
            
            if valid_steps:
                xs, ys = zip(*valid_steps)
                plt.scatter(xs, ys, color='red', marker='o', s=50, label="Các bước lặp", zorder=5)
                # Vẽ các đường tiếp tuyến
                for i in range(len(valid_steps) - 1):
                    x_curr, x_next, y_curr = valid_steps[i][0], valid_steps[i+1][0], valid_steps[i][1]
                    plt.plot([x_curr, x_curr], [0, y_curr], 'k--', lw=1, alpha=0.7)
                    plt.plot([x_curr, x_next], [y_curr, 0], 'g--', lw=1, alpha=0.7)
                    
        plt.title(f"Minh họa Newton cho khoảng ({a:.2f}, {b:.2f})")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.legend()
        plt.grid(True)
        # Đặt giới hạn trục y để đồ thị không bị "co lại" do các giá trị quá lớn
        y_min, y_max = np.min(y_vals), np.max(y_vals)
        plt.ylim([y_min - 0.2*abs(y_min), y_max + 0.2*abs(y_max)])
        
        if not os.path.exists('graphs'): os.makedirs('graphs')
        filepath = os.path.join('graphs', f'nghiem_{solution_index}.png')
        plt.savefig(filepath)
        plt.close()
        return filepath

    # --- HÀM CÔNG KHAI CHÍNH ---

    def solve(self, equation_str, user_tolerance):
        """
        Hàm công khai duy nhất để giải phương trình.
        Trả về một dictionary chứa toàn bộ kết quả.
        """
        tol = max(user_tolerance, 1e-7)
        results = {
            'success': False, 'message': '', 'analysis': {},
            'solutions': [], 'summary': {}
        }
        try:
            f = sp.sympify(equation_str)
            df = sp.diff(f, self.x)
            d2f = sp.diff(df, self.x)
            results['analysis'] = {'f_expr': str(f), 'df_expr': str(df), 'd2f_expr': str(d2f)}
        except (sp.SympifyError, TypeError):
            results['message'] = "Lỗi: Phương trình bạn nhập không hợp lệ."
            return results

        f_np, df_np, d2f_np = (sp.lambdify(self.x, expr, 'numpy') for expr in [f, df, d2f])
        
        domain = continuous_domain(f, self.x, sp.S.Reals)
        search_interval = domain.intersect(sp.Interval(-1000, 1000))
        
        if search_interval.is_empty:
             results['message'] = "Hàm số không xác định trong phạm vi quét [-1000, 1000]."
             return results

        xa = -100 if search_interval.start.is_infinite else float(search_interval.start)
        xb = 100 if search_interval.end.is_infinite else float(search_interval.end)
        
        # Xử lý biên mở
        if getattr(search_interval, 'left_open', False): xa += 1e-9
        if getattr(search_interval, 'right_open', False): xb -= 1e-9

        # Truyền thêm df symbolic vào để tìm nghiệm kép
        intervals = self._cackhoangphanly(f_np, df_np, df, xa, xb, 0.5) 
        
        if not intervals:
            results['message'] = f"Không tìm thấy khoảng phân ly nào trong phạm vi {search_interval}."
            return results

        for i, (a, b) in enumerate(intervals, 1):
            x0 = self._tim_x0(f_np, d2f_np, (a, b))
            if x0 is not None:
                nghiem, table, steps = self._pp_newton(f_np, df_np, x0, tol)
                if nghiem is not None:
                    graph_path = self._generate_and_save_graph(f_np, str(f), (a, b), steps, i)
                    results['solutions'].append({
                        'interval': (a, b), 'x0': x0, 'root': nghiem,
                        'table_str': table.get_string(), 'graph_path': graph_path
                    })
        
        if not results['solutions']:
            results['message'] = "Tìm thấy các khoảng phân ly nhưng không tìm được nghiệm hội tụ."
            return results

        unique_roots = sorted(list(set(round(s['root'], 7) for s in results['solutions'])))
        try:
            # Cố gắng tìm nghiệm chính xác để so sánh
            exact_roots_complex = sp.solve(f, self.x)
            exact_roots = [float(s) for s in exact_roots_complex if s.is_real]
        except (NotImplementedError, TypeError):
            exact_roots = "Không thể tìm nghiệm đúng bằng phương pháp giải tích."

        results['summary'] = {'approx_roots': unique_roots, 'exact_roots': exact_roots}
        results['success'] = True
        results['message'] = f"Đã tìm thấy {len(unique_roots)} nghiệm."
        return results

# =============================================================================
# PHẦN DEMO  ( XÓA ĐI KHI TÍCH HỢP)
# =============================================================================
def format_output_string(results):
    """Hàm helper bên ngoài để định dạng dictionary thành chuỗi."""
    if not results['success']:
        return f"\nLỗi: {results['message']}"
    
    output_parts = [f"\n--- PHÂN TÍCH HÀM SỐ ---\nf(x) = {results['analysis']['f_expr']}\nf'(x) = {results['analysis']['df_expr']}\nf''(x) = {results['analysis']['d2f_expr']}"]
    
    for i, sol in enumerate(results['solutions'], 1):
        output_parts.append(f"\n--- KẾT QUẢ CHO NGHIỆM {i} ---\nTìm thấy trong khoảng: ({sol['interval'][0]:.4f}, {sol['interval'][1]:.4f})\nĐiểm bắt đầu x0 = {sol['x0']:.4f}\nBảng lặp:\n{sol['table_str']}\nĐã lưu đồ thị minh họa tại: {sol['graph_path']}")
    
    summary = results['summary']
    approx_str = ", ".join([f"{r:.6f}" for r in summary['approx_roots']])
    exact_str = str(summary['exact_roots'])
    if isinstance(summary['exact_roots'], list):
        exact_str = ", ".join([f"{r:.6f}" for r in summary['exact_roots']])

    output_parts.append(f"\n--- TỔNG KẾT ---\nCác nghiệm gần đúng tìm được: {approx_str}\nCác nghiệm đúng (để tham khảo): {exact_str}")
    
    return "\n".join(output_parts)

