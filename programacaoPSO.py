import numpy as np
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


class AlgoritmoPSO:
    def __init__(self, D, S, H, C, Sseg, tam_populacao=100, num_iteracoes=100, w=0.5, c1=2, c2=2):
        self.D = D  # Demanda anual
        self.S = S  # Custo de fazer um pedido
        self.H = H  # Custo de manutenção por unidade
        self.C = C  # Capacidade máxima de armazenamento
        self.Sseg = Sseg  # Estoque de segurança
        self.tam_populacao = tam_populacao
        self.num_iteracoes = num_iteracoes
        self.w = w  # Peso de inércia
        self.c1 = c1  # Coeficiente cognitivo
        self.c2 = c2  # Coeficiente social
        self.melhor_historico = []
        self.media_historico = []

    # Faz o cálculo do custo total e aplica a penalização para soluções inviáveis
    def custo_total(self, Q):
        if Q < self.Sseg or Q > self.C:
            return float('inf')
        return (self.D / Q) * self.S + (Q / 2) * self.H

    def criar_particulas_iniciais(self):
        return np.random.uniform(self.Sseg, self.C, self.tam_populacao)

    # Avalia o custo para cada partícula
    def avaliar_particulas(self, particulas):
        return np.array([self.custo_total(Q) for Q in particulas])

    # Executa o algoritmo PSO
    def executar(self):
        # Inicialização das partículas e velocidades
        particulas = self.criar_particulas_iniciais()
        velocidades = np.zeros_like(particulas)
        pbest = particulas.copy()
        gbest = particulas[np.argmin(self.avaliar_particulas(particulas))]

        for iteracao in range(self.num_iteracoes):
            fitness = self.avaliar_particulas(particulas)

            # Armazena o melhor e a média para o histórico
            self.melhor_historico.append(np.min(fitness))
            self.media_historico.append(np.mean(fitness))

            # Atualiza pbest e gbest
            for i in range(self.tam_populacao):
                if fitness[i] < self.custo_total(pbest[i]):
                    pbest[i] = particulas[i]
                if fitness[i] < self.custo_total(gbest):
                    gbest = particulas[i]

            # Atualiza a velocidade e a posição das partículas
            for i in range(self.tam_populacao):
                r1, r2 = np.random.rand(2)
                velocidades[i] = self.w * velocidades[i] + self.c1 * r1 * (pbest[i] - particulas[i]) + self.c2 * r2 * (gbest - particulas[i])
                particulas[i] += velocidades[i]

        # Encontra a melhor solução global
        fitness_final = self.avaliar_particulas(particulas)
        melhor_idx = np.argmin(fitness_final)
        melhor_q = particulas[melhor_idx]
        melhor_custo = fitness_final[melhor_idx]

        return melhor_q, melhor_custo, self.melhor_historico, self.media_historico


class OtimizacaoEstoqueGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Otimização de Estoque - PSO")
        self.root.geometry("1200x800")

        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.pack(expand=True, fill='both')

        self.left_frame = ttk.LabelFrame(self.main_frame, text="Parâmetros", padding="10")
        self.left_frame.pack(side='left', fill='both', expand=True, padx=5)

        self.right_frame = ttk.Frame(self.main_frame)
        self.right_frame.pack(side='right', fill='both', expand=True, padx=5)

        self.criar_widgets()
        self.configurar_graficos()

    def criar_widgets(self):
        # Parâmetros do problema
        self.params = {
            'D': ('Demanda anual (D):', 1000),
            'S': ('Custo de fazer um pedido (S):', 50),
            'H': ('Custo de manutenção (H):', 2),
            'C': ('Capacidade máxima (C):', 200),
            'Sseg': ('Estoque segurança (Sseg):', 10),
            'pop': ('Tamanho da população:', 100),
            'iter': ('Número de iterações:', 100)
        }

        self.entries = {}
        for i, (key, (label, default)) in enumerate(self.params.items()):
            ttk.Label(self.left_frame, text=label).grid(row=i, column=0, sticky='w', padx=5, pady=2)
            entry = ttk.Entry(self.left_frame, width=20)
            entry.insert(0, str(default))
            entry.grid(row=i, column=1, padx=5, pady=2)
            self.entries[key] = entry

        ttk.Button(self.left_frame, text="Otimizar", command=self.otimizar).grid(row=len(self.params), column=0, columnspan=2, pady=10)

        # Área de resultado
        self.resultado_text = tk.Text(self.left_frame, height=5, width=40)
        self.resultado_text.grid(row=len(self.params)+1, column=0, columnspan=2, pady=5)

    def configurar_graficos(self):
        # Frame para os gráficos
        self.fig = Figure(figsize=(12, 5))

        # Gráfico de convergência
        self.ax1 = self.fig.add_subplot(121)
        self.ax1.set_title('Convergência do PSO')
        self.ax1.set_xlabel('Iteração')
        self.ax1.set_ylabel('Custo')

        # Gráfico de custo total
        self.ax2 = self.fig.add_subplot(122)
        self.ax2.set_title('Curva de Custo Total')
        self.ax2.set_xlabel('Quantidade de Pedido (Q)')
        self.ax2.set_ylabel('Custo Total')

        self.canvas = FigureCanvasTkAgg(self.fig, self.right_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

    def atualizar_graficos(self, melhor_q, melhor_historico, media_historico, D, S, H, C, Sseg):
        # Limpa os gráficos
        self.ax1.clear()
        self.ax2.clear()

        # Gráfico de convergência
        self.ax1.plot(melhor_historico, label='Melhor fitness')
        self.ax1.plot(media_historico, label='Média fitness')
        self.ax1.set_title('Convergência do PSO')
        self.ax1.set_xlabel('Iteração')
        self.ax1.set_ylabel('Custo')
        self.ax1.legend()
        self.ax1.grid(True)

        # Gráfico de custo total
        Q_range = np.linspace(max(1, Sseg), C, 1000)
        custos = [(D/q)*S + (q/2)*H for q in Q_range]
        self.ax2.plot(Q_range, custos, label='Custo Total')
        self.ax2.scatter([melhor_q], [(D/melhor_q)*S + (melhor_q/2)*H],
                         color='red', marker='o', s=100, label='Solução PSO')
        self.ax2.set_title('Curva de Custo Total')
        self.ax2.set_xlabel('Quantidade de Pedido (Q)')
        self.ax2.set_ylabel('Custo Total')
        self.ax2.legend()
        self.ax2.grid(True)

        self.fig.tight_layout()
        self.canvas.draw()

    def otimizar(self):
        try:
            # Coleta valores dos campos
            D = float(self.entries['D'].get())
            S = float(self.entries['S'].get())
            H = float(self.entries['H'].get())
            C = float(self.entries['C'].get())
            Sseg = float(self.entries['Sseg'].get())
            tam_pop = int(self.entries['pop'].get())
            num_iter = int(self.entries['iter'].get())

            # Cria e executa o algoritmo PSO
            pso = AlgoritmoPSO(D, S, H, C, Sseg, tam_pop, num_iter)
            melhor_q, melhor_custo, melhor_historico, media_historico = pso.executar()

            # Atualiza resultado
            resultado = (f"Melhor solução encontrada:\n"
                        f"Q* = {melhor_q:.2f}\n"
                        f"Custo total = R$ {melhor_custo:.2f}")

            self.resultado_text.delete(1.0, tk.END)
            self.resultado_text.insert(tk.END, resultado)

            # Atualiza gráficos
            self.atualizar_graficos(melhor_q, melhor_historico, media_historico, D, S, H, C, Sseg)

        except ValueError:
            messagebox.showerror("Erro", "Por favor, insira valores válidos nos campos.")

# Inicializando a interface gráfica
root = tk.Tk()
app = OtimizacaoEstoqueGUI(root)
root.mainloop()
