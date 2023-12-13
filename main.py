from collections import defaultdict
import random
import math

def mod_pow(a, x, p):
    res = 1
    a = a % p
    if a == 0:
        return 0
    while x > 0:
        if x & 1 == 1:
            res = (res * a) % p
        a = (a ** 2) % p
        x >>= 1
    return res

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def extended_gcd(a, b):
    U = (a, 1, 0)
    V = (b, 0, 1)
    while V[0] != 0:
        q = U[0] // V[0]
        T = (U[0] % V[0], U[1] - q * V[1], U[2] - q * V[2])
        U = V
        V = T
    return U

def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

def get_prime(start, end):
    while True:
        p = random.randint(start, end)
        if is_prime(p):
            return p

def get_coprime(p):
    res = random.randint(2, p)
    while gcd(p, res) != 1:
        res = random.randint(2, p)
    return res

def create_isomorphic_graph(graph, n):
    new_vertices = list(range(1, n + 1))
    random.shuffle(new_vertices)
    vertex_map = {i: new_vertices[i - 1] for i in range(1, n + 1)}
    new_graph = [[0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if graph[i][j] == 1:
                new_i, new_j = vertex_map[i + 1], vertex_map[j + 1]
                new_graph[new_i - 1][new_j - 1] = 1

    return new_graph, vertex_map

def get_RSA():
    P = get_prime(10**6, 10**9)
    Q = get_prime(10**6, 10**9)
    N = P * Q

    phi = (P - 1) * (Q - 1)

    d = get_coprime(phi)

    c = extended_gcd(d, phi)[1]
    if c < 0:
        c += phi
    
    print("P = ", P)
    print("Q = ", Q)
    print("N = ", N)
    print("phi = ", phi)
    print("c = ", c)
    return N, d

def RSA_encode(graph, x, p, n):
    temp_graph = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            temp_graph[i][j] = mod_pow(graph[i][j], x, p)
    return temp_graph

def build_adjacency_list(graph):
    adjacency_list = defaultdict(list)
    for row_index, row in enumerate(graph):
        for col_index, value in enumerate(row):
            if value == 1:
                adjacency_list[row_index].append(col_index + 1)
    return adjacency_list

def find_hamiltonian_cycle(graph, start_vertex, path=[]):
    if len(path) == 0:
        path.append(start_vertex)
    
    if len(path) == len(graph):
        if path[0] in graph[path[-1] - 1]:
            return path + [path[0]]  # Добавляем стартовую вершину в конец для замыкания цикла
        else:
            return None

    for vertex in graph[path[-1] - 1]:
        if vertex not in path:
            next_path = path + [vertex]
            cycle = find_hamiltonian_cycle(graph, start_vertex, next_path)
            if cycle:
                return cycle

    return None

def check_graph_equality(graph1, graph2):
    return all(graph1[i][j] == graph2[i][j] for i in range(len(graph1)) for j in range(len(graph2[i])))


def print_graph(graph, n):
    print("—" * (9 * n - 1))
    print(" " * 6 + "|\t", end="")
    print("\t".join(str(i) for i in range(1, n + 1)))
    print("—" * (9 * n - 1))

    for i in range(n):
        print(str(i + 1).ljust(6) + "|\t", end="")
        for j in range(n):
            print(graph[i][j], end="\t")
        print("\n", end="")
    print("\n")
    
def print_graph_RSA(graph, n):
    print("—" * (24 * n))
    print(" " * 6 + "|\t", end="")
    print("\t\t\t".join(str(i) for i in range(1, n + 1)))
    print("—" * (24 * n))

    for i in range(n):
        print(str(i + 1).ljust(6) + "|\t", end="")
        for j in range(n):
            print(graph[i][j], end="\t")
        print("\n", end="")
    print("\n")

def main():
    try:
        file_path = r"C:\vsCode\InfoProtection\RGR\graph.txt"
        with open(file_path, 'r') as f:
            rows = f.read().splitlines()
    except OSError as e:
        print(f"Ошибка открытия файла с графами: {e}")
        return

    # Извлечение n и m из первой строки
    n, m = map(int, rows[0].split())

    Graph_G = [[0 for _ in range(n)] for _ in range(n)]
    Graph_H = [[0 for _ in range(n)] for _ in range(n)]
    Graph_H_ = [[0 for _ in range(n)] for _ in range(n)]
    Graph_F = [[0 for _ in range(n)] for _ in range(n)]

    # заполнение графа Алисой
    for k in range(m):
        i, j = map(int, rows[k + 1].split())
        Graph_G[i - 1][j - 1] = 1
        Graph_G[j - 1][i - 1] = 1

    print("Заполненный граф G:")
    print_graph(Graph_G, n)
    
    # строим список смежности
    adj_list1 = build_adjacency_list(Graph_G)
    print("Список смежности для графа G:")
    for key, value in adj_list1.items():
        print(f"{key + 1}: {value}")

    # ищем гамильтонов цикл
    hamiltonian_cycle1 = find_hamiltonian_cycle(adj_list1, 1)
    print(f"\nГамильтонов цикл: {hamiltonian_cycle1}\n")


    Graph_H, vertex_map = create_isomorphic_graph(Graph_G, n)
    print("Новая нумерация вершин для изоморфного графа H:")
    for original, new in vertex_map.items():
        print(f"{original} -> {new}")

    print("\nИзоморфный граф H:")
    print_graph(Graph_H, n)

    print("Граф H_:")
    for i in range(n):
        for j in range(n):
            random_number = random.randint(1, 9)
            Graph_H_[i][j] = int(str(random_number) + str(Graph_H[i][j]))
            
    print_graph(Graph_H_, n)

    print("Алиса генерирует ключи:")
    N, d = get_RSA()

    Graph_F = RSA_encode(Graph_H_, d, N, n)
    print("\nГраф F для Боба после шифра RSA:")
    print_graph_RSA(Graph_F, n)
    
    print("1. Алиса, каков Гамильтонов цикл для графа H?")
    print("2. Алиса, действительно ли граф H изоморфен G?")
    question = int(input("\nВыберите вопрос: "))

    if question == 1:
        Bob_Graph_F = Graph_F
        print("Алиса посылает Бобу список ребер графа H_")

        adj_list2 = build_adjacency_list(Graph_H)
        print("Список смежности для графа H:")
        for key, value in adj_list2.items():
            print(f"{key + 1}: {value}")

        hamiltonian_cycle2 = find_hamiltonian_cycle(adj_list2, 1)
        print(f"\nГамильтонов цикл: {hamiltonian_cycle2}\n")

        # формирование списка ребер гамильтонова цикла
        edges = [(hamiltonian_cycle2[i], hamiltonian_cycle2[i + 1]) for i in range(len(hamiltonian_cycle2) - 1)]

        # получение значения каждого ребра из графа H_
        edges_with_values = []
        for edge in edges:
            i, j = edge
            edge_value = Graph_H_[i - 1][j - 1]
            edges_with_values.append((i, j, edge_value))

        # вывод списка ребер с их значениями
        print("Список ребер гамильтонова цикла с их значениями в графе H_:")
        for edge in edges_with_values:
            print(edge)

        # проверка соответствия ребер графу F
        for edge in edges_with_values:
            i, j, edge_value = edge
            encrypted_value = mod_pow(edge_value, d, N)
            # print(encrypted_value)
            if Bob_Graph_F[i - 1][j - 1] != encrypted_value:
                print(f"Несоответствие обнаружено: ребро ({i}, {j}) со значением {edge_value} не соответствует матрице F.")
                break
        else:
            print("Все ребра соответствуют матрице F.")

        # проверка, что путь проходит через все вершины по одному разу
        visited_vertices = set()
        for edge in edges_with_values:
            i, j, _ = edge
            visited_vertices.add(i)
            visited_vertices.add(j)

        if len(visited_vertices) == len(Bob_Graph_F):
            print("Путь проходит через все вершины графа по одному разу.")
        else:
            print("Путь не проходит через все вершины графа по одному разу.")

    if question == 2:
        print("Боб получает матрицу F и запрашивает доказательство изоморфизма.")

        print("Алиса посылает закодированную матрицу H_ и нумерацию вершин.")
        print("Закодированная матрица H_:")
        print_graph(Graph_H_, n)
        print("Перестановки, с помощью которых граф H был получен из графа G:")
        vertex_mapping_str = " ".join(str(new) for _, new in vertex_map.items())
        print(vertex_mapping_str)

        print("\nБоб проверяет соответствие графа H_ графу F:")
        for i in range(n):
            for j in range(n):
                if mod_pow(Graph_H_[i][j], d, N) != Graph_F[i][j]:
                    print(f"Несоответствие в элементе ({i+1}, {j+1}).")
                    break
            else:
                continue
            break
        else:
            print("Граф H_ соответствует графу F.")

        print("\nБоб получает граф H из матрицы H_:")
        Bob_Graph_H = [[int(str(Graph_H_[i][j])[-1]) for j in range(n)] for i in range(n)]
        print_graph(Bob_Graph_H, n)

        print("Боб переставляет вершины в графе G согласно полученной нумерации:")
        Bob_Graph_G_to_H = [[0 for _ in range(n)] for _ in range(n)]
        for old_i, new_i in vertex_map.items():
            for old_j, new_j in vertex_map.items():
                Bob_Graph_G_to_H[new_i - 1][new_j - 1] = Graph_G[old_i - 1][old_j - 1]

        print("Граф H:")
        print_graph(Bob_Graph_H, n)
        
        print("Граф G после перестановок:")
        print_graph(Bob_Graph_G_to_H, n)
        
        print("Сравнение графов H и G:")
        if check_graph_equality(Bob_Graph_H, Bob_Graph_G_to_H):
            print("Графы H и G идентичны, следовательно, они изоморфны.")
        else:
            print("Графы H и G не идентичны, следовательно, они не изоморфны.")
    
    return 0


if __name__ == "__main__":
    main()
