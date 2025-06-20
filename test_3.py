import tkinter as tk
import folium
import json
import csv
import math
import heapq
import networkx as nx
import matplotlib.pyplot as plt
from tkinter import ttk, messagebox, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import defaultdict
from folium.plugins import MarkerCluster
import webbrowser
import os
from abc import ABC, abstractmethod
from folium.plugins import AntPath
import requests
import threading
from queue import Queue
from typing import List, Dict, Optional, Tuple, Any


class NodeManager:
    """Управляет узлами (пунктами доставки)"""

    def __init__(self):
        self.nodes = []
        self.next_id = 1

    def add_node(self, name: str, latitude: float, longitude: float) -> int:
        """Добавляет новый узел и возвращает его ID"""
        node_id = self.next_id
        self.nodes.append({
            'id': node_id,
            'name': name,
            'latitude': latitude,
            'longitude': longitude
        })
        self.next_id += 1
        return node_id

    def get_node_by_id(self, node_id: int) -> Optional[Dict]:
        """Возвращает узел по ID или None если не найден"""
        return next((node for node in self.nodes if node['id'] == node_id), None)

    def clear_nodes(self):
        """Очищает все узлы и сбрасывает счетчик ID"""
        self.nodes = []
        self.next_id = 1

    def get_nodes_data(self) -> List[Dict]:
        """Возвращает копию данных о всех узлах"""
        return [node.copy() for node in self.nodes]


class EdgeManager:
    """Управляет ребрами (связями между пунктами)"""

    def __init__(self, node_manager: NodeManager):
        self.edges = []
        self.node_manager = node_manager
        self.routing_service_url = "http://router.project-osrm.org/route/v1/driving/"
        self.route_cache = {}
        self.request_queue = Queue()
        self.results = {}
        self.worker_thread = None
        self.avg_speed = 50.0  # км/ч
        self.lock = threading.Lock()
        self.stop_event = threading.Event()

    def _get_cache_key(self, node1: Dict, node2: Dict) -> str:
        """Генерирует ключ для кэша на основе координат узлов"""
        coords = sorted([
            (node1['longitude'], node1['latitude']),
            (node2['longitude'], node2['latitude'])
        ])
        return f"{coords[0][0]},{coords[0][1]};{coords[1][0]},{coords[1][1]}"

    @staticmethod
    def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Вычисляет расстояние между двумя точками на сфере (в км)"""
        R = 6371.0  # Радиус Земли в км

        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)

        dlon = lon2_rad - lon1_rad
        dlat = lat2_rad - lat1_rad

        a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    def _get_route_distance_and_duration(self, lon1: float, lat1: float, lon2: float, lat2: float) -> Tuple[
        float, float]:
        """Получает расстояние (км) и время (сек) маршрута по дорогам с помощью OSRM API"""
        cache_key = self._get_cache_key(
            {'longitude': lon1, 'latitude': lat1},
            {'longitude': lon2, 'latitude': lat2}
        )

        if cache_key in self.route_cache:
            return self.route_cache[cache_key]

        url = f"{self.routing_service_url}{lon1},{lat1};{lon2},{lat2}?overview=false"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data.get('code') == 'Ok' and data.get('routes'):
                route = data['routes'][0]
                result = (route['distance'] / 1000, route['duration'])  # Переводим метры в км
                with self.lock:
                    self.route_cache[cache_key] = result
                return result
            raise ValueError(f"OSRM API error: {data.get('message', 'Unknown error')}")
        except (requests.RequestException, ValueError, KeyError) as e:
            print(f"OSRM API request failed: {str(e)}")
            # В случае ошибки возвращаем расстояние по прямой
            distance = self._haversine_distance(lat1, lon1, lat2, lon2)
            return distance, distance / self.avg_speed * 3600  # Предполагаем время в секундах

    def calculate_edges(self, avg_speed: float = 50.0):
        """Рассчитывает связи между всеми узлами"""
        self.avg_speed = avg_speed
        self.edges = []
        nodes = self.node_manager.nodes

        if len(nodes) < 2:
            return

        # Сначала заполняем кэш известными значениями
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                node1 = nodes[i]
                node2 = nodes[j]
                cache_key = self._get_cache_key(node1, node2)

                if cache_key in self.route_cache:
                    distance, duration = self.route_cache[cache_key]
                    time_val = duration / 3600  # Переводим секунды в часы
                    self.edges.append({
                        'from': node1['id'],
                        'to': node2['id'],
                        'distance': distance,
                        'time': time_val
                    })
                else:
                    # Для некэшированных пар используем расстояние по прямой
                    distance = self._haversine_distance(
                        node1['latitude'], node1['longitude'],
                        node2['latitude'], node2['longitude']
                    )
                    time_val = distance / avg_speed
                    self.edges.append({
                        'from': node1['id'],
                        'to': node2['id'],
                        'distance': distance,
                        'time': time_val
                    })

        # Затем асинхронно обновляем данные через API
        self._update_edges_via_api()

    def _update_edges_via_api(self):
        """Асинхронно обновляет расстояния через API"""
        nodes = self.node_manager.nodes

        # Останавливаем предыдущий worker, если он работает
        self.stop_event.set()
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1)
        self.stop_event.clear()

        # Создаем и запускаем новый worker-поток
        self.worker_thread = threading.Thread(target=self._process_requests)
        self.worker_thread.daemon = True
        self.worker_thread.start()

        # Добавляем все пары узлов в очередь для обработки
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                node1 = nodes[i]
                node2 = nodes[j]
                cache_key = self._get_cache_key(node1, node2)

                if cache_key not in self.route_cache:
                    self.request_queue.put(((node1['id'], node2['id']), node1, node2))

        # Запускаем проверку завершения
        threading.Thread(target=self._check_completion, args=(len(nodes),)).start()

    def _process_requests(self):
        """Обрабатывает запросы в фоновом потоке"""
        while not self.stop_event.is_set():
            try:
                key, node1, node2 = self.request_queue.get(timeout=1)

                try:
                    distance, duration = self._get_route_distance_and_duration(
                        node1['longitude'], node1['latitude'],
                        node2['longitude'], node2['latitude']
                    )
                    time_val = duration / 3600
                except Exception as e:
                    print(f"Ошибка расчета расстояния: {str(e)}")
                    distance = self._haversine_distance(
                        node1['latitude'], node1['longitude'],
                        node2['latitude'], node2['longitude']
                    )
                    time_val = distance / self.avg_speed

                # Обновляем результаты
                with self.lock:
                    self.results[key] = {
                        'from': node1['id'],
                        'to': node2['id'],
                        'distance': distance,
                        'time': time_val
                    }

                self.request_queue.task_done()
            except:
                continue

    def _check_completion(self, total_nodes: int):
        """Проверяет завершение всех расчетов"""
        expected_pairs = (total_nodes * (total_nodes - 1)) // 2

        while len(self.results) < expected_pairs and not self.stop_event.is_set():
            threading.Event().wait(0.1)

        if self.stop_event.is_set():
            return

        # Все расчеты завершены, обновляем ребра
        with self.lock:
            self.edges = [edge for edge in self.edges if (edge['from'], edge['to']) not in self.results]
            self.edges.extend(self.results.values())
            self.results.clear()

        # Вызываем обновление графа в основном потоке
        if hasattr(self.node_manager, 'app'):
            self.node_manager.app.root.after(0, self._update_graph)

    def _update_graph(self):
        """Обновляет граф в основном потоке"""
        if hasattr(self.node_manager, 'app'):
            self.node_manager.app.graph_manager.update_graph()
            self.node_manager.app._update_graph()

    def clear_edges(self):
        """Очищает все ребра и кэш"""
        self.edges = []
        self.route_cache = {}
        self.results = {}
        self.stop_event.set()


class GraphManager:
    """Управляет графом и алгоритмами маршрутизации"""

    def __init__(self, node_manager: NodeManager, edge_manager: EdgeManager):
        self.graph = nx.Graph()
        self.node_manager = node_manager
        self.edge_manager = edge_manager

    def update_graph(self):
        """Обновляет граф на основе текущих узлов и ребер"""
        self.graph.clear()

        # Добавляем узлы
        for node in self.node_manager.nodes:
            self.graph.add_node(
                node['id'],
                pos=(node['longitude'], node['latitude']),
                label=node['name']
            )

        # Добавляем ребра
        for edge in self.edge_manager.edges:
            self.graph.add_edge(
                edge['from'],
                edge['to'],
                weight=edge['distance'],
                time=edge['time']
            )

    def find_path(self, start_id: int, end_id: int, algorithm: str = 'dijkstra') -> Optional[List[int]]:
        """Находит путь между двумя узлами с использованием указанного алгоритма"""
        if not self.graph.has_node(start_id) or not self.graph.has_node(end_id):
            return None

        if algorithm == 'dijkstra':
            try:
                return nx.dijkstra_path(self.graph, start_id, end_id, weight='weight')
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                return None
        elif algorithm == 'astar':
            return self._astar_algorithm(start_id, end_id)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    def _astar_algorithm(self, start: int, goal: int) -> Optional[List[int]]:
        """Реализация алгоритма A*"""
        if not self.graph.has_node(start) or not self.graph.has_node(goal):
            return None

        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = defaultdict(lambda: float('inf'))
        g_score[start] = 0
        f_score = defaultdict(lambda: float('inf'))
        f_score[start] = self._heuristic(start, goal)

        open_set_hash = {start}

        while open_set:
            current = heapq.heappop(open_set)[1]
            open_set_hash.remove(current)

            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path

            for neighbor in self.graph.neighbors(current):
                tentative_g_score = g_score[current] + self.graph[current][neighbor]['weight']

                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self._heuristic(neighbor, goal)
                    if neighbor not in open_set_hash:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
                        open_set_hash.add(neighbor)

        return None

    def _heuristic(self, node1: int, node2: int) -> float:
        """Эвристическая функция для A* (расстояние по прямой)"""
        node1_data = self.node_manager.get_node_by_id(node1)
        node2_data = self.node_manager.get_node_by_id(node2)

        if not node1_data or not node2_data:
            return float('inf')

        return self.edge_manager._haversine_distance(
            node1_data['latitude'], node1_data['longitude'],
            node2_data['latitude'], node2_data['longitude']
        )


class RouteCalculator(ABC):
    """Абстрактный класс для расчета маршрутов"""

    @abstractmethod
    def calculate_route(self, start_id: int, end_id: int) -> Optional[List[int]]:
        pass


class DijkstraRouteCalculator(RouteCalculator):
    """Реализация расчета маршрута с использованием алгоритма Дейкстры"""

    def __init__(self, graph_manager: GraphManager):
        self.graph_manager = graph_manager

    def calculate_route(self, start_id: int, end_id: int) -> Optional[List[int]]:
        return self.graph_manager.find_path(start_id, end_id, 'dijkstra')


class AStarRouteCalculator(RouteCalculator):
    """Реализация расчета маршрута с использованием алгоритма A*"""

    def __init__(self, graph_manager: GraphManager):
        self.graph_manager = graph_manager

    def calculate_route(self, start_id: int, end_id: int) -> Optional[List[int]]:
        return self.graph_manager.find_path(start_id, end_id, 'astar')


class RouteCalculatorFactory:
    """Фабрика для создания калькуляторов маршрутов"""

    @staticmethod
    def create_calculator(algorithm: str, graph_manager: GraphManager) -> RouteCalculator:
        if algorithm == 'dijkstra':
            return DijkstraRouteCalculator(graph_manager)
        elif algorithm == 'astar':
            return AStarRouteCalculator(graph_manager)
        raise ValueError(f"Unknown algorithm: {algorithm}")


class DataExporter(ABC):
    """Абстрактный класс для экспорта данных"""

    @abstractmethod
    def export(self, data: Any, file_path: str) -> None:
        pass


class CSVExporter(DataExporter):
    """Экспорт данных в CSV"""

    def export(self, data: List[List[Any]], file_path: str) -> None:
        with open(file_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(data)


class JSONExporter(DataExporter):
    """Экспорт данных в JSON"""

    def export(self, data: Dict, file_path: str) -> None:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


class Visualizer:
    """Отвечает за визуализацию данных"""

    def __init__(self, node_manager: NodeManager, edge_manager: EdgeManager, graph_manager: GraphManager):
        self.node_manager = node_manager
        self.edge_manager = edge_manager
        self.graph_manager = graph_manager
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.current_route = []
        self.map_type = 'OpenStreetMap'
        self.fig.patch.set_facecolor('#f0f0f0')

    def update_visualization(self, current_route: Optional[List[int]] = None) -> None:
        """Обновляет визуализацию графа"""
        if current_route:
            self.current_route = current_route

        self.ax.clear()

        if not self.graph_manager.graph.nodes:
            self.ax.text(0.5, 0.5, 'Нет данных для отображения',
                         ha='center', va='center', fontsize=12)
            return

        pos = nx.get_node_attributes(self.graph_manager.graph, 'pos')
        labels = nx.get_node_attributes(self.graph_manager.graph, 'label')

        # Рисуем весь граф
        nx.draw(
            self.graph_manager.graph, pos, ax=self.ax,
            with_labels=True, labels=labels,
            node_size=300, node_color='skyblue',
            font_size=8, font_weight='bold',
            width=1.5, edge_color='gray', alpha=0.7
        )

        # Если есть текущий маршрут, выделяем его
        if self.current_route:
            route_edges = [
                (self.current_route[i], self.current_route[i + 1])
                for i in range(len(self.current_route) - 1)
            ]

            # Рисуем узлы маршрута
            nx.draw_networkx_nodes(
                self.graph_manager.graph, pos,
                nodelist=self.current_route,
                node_color='red', node_size=300, ax=self.ax
            )

            # Рисуем ребра маршрута
            nx.draw_networkx_edges(
                self.graph_manager.graph, pos,
                edgelist=route_edges,
                edge_color='red', width=3, ax=self.ax, alpha=0.8
            )

            # Добавляем подписи расстояний
            route_edge_labels = {
                edge: f"{self.graph_manager.graph.edges[edge]['weight']:.1f} км"
                for edge in route_edges
                if self.graph_manager.graph.has_edge(*edge)
            }

            nx.draw_networkx_edge_labels(
                self.graph_manager.graph, pos,
                edge_labels=route_edge_labels,
                font_size=7, ax=self.ax,
                font_color='red',
                bbox=dict(boxstyle='round', alpha=0.8, facecolor='white', edgecolor='none')
            )

        # Настройка осей и заголовков
        self.ax.set_title('Граф пунктов доставки', fontsize=10, pad=12)
        self.ax.set_xlabel('Долгота', fontsize=8)
        self.ax.set_ylabel('Широта', fontsize=8)
        self.ax.tick_params(axis='both', which='major', labelsize=7, pad=2)
        self.ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
        self.ax.autoscale_view()
        self.fig.tight_layout()

    def create_map_visualization(self, current_route: List[int], route_info: Dict) -> folium.Map:
        """Создает интерактивную карту с маршрутом по дорогам"""
        if not self.node_manager.nodes:
            raise ValueError("Нет данных для визуализации")

        first_node = self.node_manager.nodes[0]

        # Создаем карту с выбранным типом
        if self.map_type == 'Satellite':
            tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'
            attr = 'Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community'
            m = folium.Map(
                location=[first_node['latitude'], first_node['longitude']],
                zoom_start=12,
                tiles=tiles,
                attr=attr
            )
        else:
            m = folium.Map(
                location=[first_node['latitude'], first_node['longitude']],
                zoom_start=12,
                control_scale=True
            )

        # Добавляем все пункты на карту
        marker_cluster = MarkerCluster().add_to(m)
        for node in self.node_manager.nodes:
            folium.Marker(
                location=[node['latitude'], node['longitude']],
                popup=f"{node['name']} (ID: {node['id']})",
                icon=folium.Icon(color='blue' if node['id'] not in current_route else 'red')
            ).add_to(marker_cluster)

        # Если есть маршрут, получаем его по дорогам с помощью OSRM
        if current_route and len(current_route) >= 2:
            try:
                # Получаем координаты всех точек маршрута
                coords = []
                for node_id in current_route:
                    node = self.node_manager.get_node_by_id(node_id)
                    if node:
                        coords.append(f"{node['longitude']},{node['latitude']}")

                # Формируем запрос к OSRM
                url = f"http://router.project-osrm.org/route/v1/driving/{';'.join(coords)}?overview=full&geometries=geojson"

                # Отправляем запрос
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                data = response.json()

                if data.get('code') == 'Ok' and data.get('routes'):
                    # Получаем геометрию маршрута
                    geometry = data['routes'][0]['geometry']
                    coordinates = geometry['coordinates']

                    # Преобразуем координаты для Folium
                    route_coords = [[lat, lon] for lon, lat in coordinates]

                    # Рисуем маршрут
                    folium.PolyLine(
                        locations=route_coords,
                        color='red',
                        weight=5,
                        opacity=0.8,
                        tooltip=f"Маршрут: {' → '.join(route_info['route_names'])}"
                    ).add_to(m)

                    # Добавляем маркеры начала и конца
                    start_node = self.node_manager.get_node_by_id(current_route[0])
                    end_node = self.node_manager.get_node_by_id(current_route[-1])

                    if start_node:
                        folium.Marker(
                            location=[start_node['latitude'], start_node['longitude']],
                            popup=f"Начало: {start_node['name']}",
                            icon=folium.Icon(color='green')
                        ).add_to(m)

                    if end_node:
                        folium.Marker(
                            location=[end_node['latitude'], end_node['longitude']],
                            popup=f"Конец: {end_node['name']}",
                            icon=folium.Icon(color='darkred')
                        ).add_to(m)
                else:
                    self._draw_straight_line(m, current_route, route_info)

            except (requests.RequestException, ValueError, KeyError) as e:
                print(f"Ошибка при построении маршрута: {str(e)}")
                self._draw_straight_line(m, current_route, route_info)

        return m

    def _draw_straight_line(self, m: folium.Map, current_route: List[int], route_info: Dict) -> None:
        """Рисует прямую линию маршрута (резервный вариант)"""
        route_coords = []
        for node_id in current_route:
            node = self.node_manager.get_node_by_id(node_id)
            if node:
                route_coords.append([node['latitude'], node['longitude']])

        AntPath(
            locations=route_coords,
            color='red',
            weight=5,
            opacity=0.8,
            tooltip=f"Маршрут: {' → '.join(route_info['route_names'])}",
            dash_array=[10, 20]
        ).add_to(m)


class RouteOptimizer:
    """Координирует процесс оптимизации маршрутов"""

    def __init__(self, node_manager: NodeManager, edge_manager: EdgeManager,
                 graph_manager: GraphManager, route_calculator: RouteCalculator):
        self.node_manager = node_manager
        self.edge_manager = edge_manager
        self.graph_manager = graph_manager
        self.route_calculator = route_calculator
        self.current_route = []

    def calculate_route(self, selected_nodes: List[int]) -> Dict:
        """Рассчитывает оптимальный маршрут через выбранные узлы"""
        if len(selected_nodes) < 2:
            raise ValueError("Выберите хотя бы два пункта")

        total_distance = 0.0
        total_time = 0.0
        full_path = []
        route_names = []

        # Рассчитываем маршрут между последовательными выбранными пунктами
        for i in range(len(selected_nodes) - 1):
            start = selected_nodes[i]
            end = selected_nodes[i + 1]

            path = self.route_calculator.calculate_route(start, end)
            if not path:
                raise ValueError(f"Не удалось найти путь между узлами {start} и {end}")

            # Рассчитываем расстояние и время для этого участка
            distance = self._calculate_path_distance(path)
            time_val = self._calculate_path_time(path)

            # Добавляем путь (исключая последний узел, чтобы избежать дублирования)
            full_path.extend(path[:-1])
            total_distance += distance
            total_time += time_val

        # Добавляем последний узел
        full_path.append(selected_nodes[-1])
        self.current_route = full_path

        # Формируем список названий пунктов
        for node_id in full_path:
            node = self.node_manager.get_node_by_id(node_id)
            route_names.append(node['name'] if node else str(node_id))

        return {
            'distance': total_distance,
            'time': total_time,
            'route': full_path,
            'route_names': route_names
        }

    def _calculate_path_distance(self, path: List[int]) -> float:
        """Вычисляет общее расстояние для пути по дорогам (в км)"""
        total_distance = 0.0

        for i in range(len(path) - 1):
            start_node = self.node_manager.get_node_by_id(path[i])
            end_node = self.node_manager.get_node_by_id(path[i + 1])

            if not start_node or not end_node:
                continue

            try:
                # Получаем расстояние по дорогам между двумя точками
                distance, _ = self.edge_manager._get_route_distance_and_duration(
                    start_node['longitude'], start_node['latitude'],
                    end_node['longitude'], end_node['latitude']
                )
                total_distance += distance
            except:
                # В случае ошибки используем расстояние из ребра графа
                edge = next(
                    (e for e in self.edge_manager.edges
                     if (e['from'] == path[i] and e['to'] == path[i + 1]) or
                     (e['from'] == path[i + 1] and e['to'] == path[i])),
                    None
                )
                if edge:
                    total_distance += edge['distance']

        return total_distance

    def _calculate_path_time(self, path: List[int]) -> float:
        """Вычисляет общее время для пути (в часах)"""
        total_time = 0.0

        for i in range(len(path) - 1):
            edge = next(
                (e for e in self.edge_manager.edges
                 if (e['from'] == path[i] and e['to'] == path[i + 1]) or
                 (e['from'] == path[i + 1] and e['to'] == path[i])),
                None
            )
            if edge:
                total_time += edge['time']

        return total_time


class DeliveryOptimizerApp:
    """Главный класс приложения"""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title('Оптимизатор маршрутов доставки')
        self.root.geometry('1200x800')
        self.root.minsize(800, 600)

        # Инициализация стиля
        self._init_styles()

        # Инициализация менеджеров
        self.node_manager = NodeManager()
        self.edge_manager = EdgeManager(self.node_manager)
        self.graph_manager = GraphManager(self.node_manager, self.edge_manager)
        self.visualizer = Visualizer(self.node_manager, self.edge_manager, self.graph_manager)

        # Добавляем ссылку на приложение в менеджеры
        self.node_manager.app = self
        self.edge_manager.node_manager.app = self

        # Инициализация интерфейса
        self._init_ui()

        # Холст для графа
        self.canvas = FigureCanvasTkAgg(self.visualizer.fig, master=self.visualization_tab)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Состояние приложения
        self.last_map_file = None

    def _init_styles(self):
        """Инициализация стилей для виджетов"""
        style = ttk.Style()
        style.configure('TFrame', background='#f0f0f0')
        style.configure('TLabel', background='#f0f0f0', font=('Arial', 9))
        style.configure('TButton', font=('Arial', 9))
        style.configure('TEntry', font=('Arial', 9))
        style.configure('Treeview', font=('Arial', 9), rowheight=25)
        style.configure('Treeview.Heading', font=('Arial', 9, 'bold'))
        style.configure('TCombobox', font=('Arial', 9))
        style.configure('TNotebook.Tab', font=('Arial', 9, 'bold'))

    def _init_ui(self):
        """Инициализация пользовательского интерфейса"""
        # Переменные
        self.algorithm_var = tk.StringVar(value='dijkstra')
        self.auto_calc_var = tk.BooleanVar(value=True)
        self.avg_speed_var = tk.StringVar(value='50')
        self.distance_var = tk.StringVar(value='0 км')
        self.time_var = tk.StringVar(value='0 ч')
        self.route_var = tk.StringVar(value='')
        self.max_load_var = tk.StringVar()
        self.max_time_var = tk.StringVar()
        self.map_type_var = tk.StringVar(value='OpenStreetMap')

        # Создание вкладок
        self.notebook = ttk.Notebook(self.root)
        self.data_tab = ttk.Frame(self.notebook)
        self.route_tab = ttk.Frame(self.notebook)
        self.visualization_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.data_tab, text='Данные')
        self.notebook.add(self.route_tab, text='Рассчитать маршрут')
        self.notebook.add(self.visualization_tab, text='Визуализация')
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Инициализация компонентов
        self._init_data_tab()
        self._init_route_tab()
        self._init_visualization_tab()

    def _init_data_tab(self):
        """Инициализация вкладки данных"""
        data_frame = ttk.LabelFrame(self.data_tab, text='Управление данными', padding=10)
        data_frame.pack(fill=tk.X, padx=10, pady=10)

        btn_frame = ttk.Frame(data_frame)
        btn_frame.pack(fill=tk.X)

        ttk.Button(btn_frame, text='Добавить пункт', command=self._add_node_dialog).pack(side=tk.LEFT, padx=5)

        # Меню для операций с файлами
        file_menu = tk.Menubutton(btn_frame, text='Файлы', relief=tk.RAISED)
        file_menu.pack(side=tk.LEFT, padx=5)
        file_menu.menu = tk.Menu(file_menu, tearoff=0)
        file_menu['menu'] = file_menu.menu

        # Подменю для загрузки
        load_menu = tk.Menu(file_menu.menu, tearoff=0)
        file_menu.menu.add_cascade(label='Загрузить', menu=load_menu)
        load_menu.add_command(label='Из CSV', command=self._load_csv)
        load_menu.add_command(label='Из JSON', command=self._load_json)

        # Подменю для сохранения
        save_menu = tk.Menu(file_menu.menu, tearoff=0)
        file_menu.menu.add_cascade(label='Сохранить', menu=save_menu)
        save_menu.add_command(label='В CSV', command=self._save_csv)
        save_menu.add_command(label='В JSON', command=self._save_json)

        ttk.Button(btn_frame, text='Очистить всё', command=self._clear_all).pack(side=tk.LEFT, padx=5)

        # Таблица пунктов
        nodes_frame = ttk.LabelFrame(self.data_tab, text='Пункты доставки', padding=10)
        nodes_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.nodes_tree = ttk.Treeview(nodes_frame, columns=('id', 'name', 'latitude', 'longitude'), show='headings')
        self.nodes_tree.heading('id', text='ID')
        self.nodes_tree.heading('name', text='Название')
        self.nodes_tree.heading('latitude', text='Широта')
        self.nodes_tree.heading('longitude', text='Долгота')

        # Настройка колонок
        self.nodes_tree.column('id', width=50, anchor=tk.CENTER)
        self.nodes_tree.column('name', width=200, anchor=tk.W)
        self.nodes_tree.column('latitude', width=100, anchor=tk.CENTER)
        self.nodes_tree.column('longitude', width=100, anchor=tk.CENTER)

        self.nodes_tree.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(nodes_frame, orient='vertical', command=self.nodes_tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.nodes_tree.configure(yscrollcommand=scrollbar.set)

        # Прогресс-бар
        self.progress_frame = ttk.Frame(self.data_tab)
        self.progress_frame.pack(fill=tk.X, padx=10, pady=5)

        self.progress_label = ttk.Label(self.progress_frame, text="Готово")
        self.progress_label.pack(side=tk.LEFT, padx=5)

        self.progress = ttk.Progressbar(self.progress_frame, mode='determinate')
        self.progress.pack(fill=tk.X, expand=True, padx=5)

    def _init_route_tab(self):
        """Инициализация вкладки расчета маршрута"""
        main_frame = ttk.Frame(self.route_tab)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Левая панель
        left_panel = ttk.Frame(main_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        # Выбор пунктов
        nodes_frame = ttk.LabelFrame(left_panel, text='Выберите пункты', padding=10)
        nodes_frame.pack(fill=tk.X, padx=5, pady=5)

        self.node_listbox = tk.Listbox(nodes_frame, selectmode=tk.MULTIPLE, height=10, width=30)
        self.node_listbox.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(nodes_frame, orient='vertical', command=self.node_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.node_listbox.config(yscrollcommand=scrollbar.set)

        # Параметры расчета
        params_frame = ttk.LabelFrame(left_panel, text='Параметры расчета', padding=10)
        params_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(params_frame, text='Алгоритм:').pack(anchor=tk.W)
        algorithm_combo = ttk.Combobox(params_frame, textvariable=self.algorithm_var,
                                       values=['dijkstra', 'astar'], state='readonly')
        algorithm_combo.pack(fill=tk.X, pady=2)

        ttk.Label(params_frame, text='Средняя скорость (км/ч):').pack(anchor=tk.W)
        ttk.Entry(params_frame, textvariable=self.avg_speed_var).pack(fill=tk.X, pady=2)

        ttk.Checkbutton(params_frame, text='Автоматический пересчет', variable=self.auto_calc_var).pack(anchor=tk.W,
                                                                                                        pady=5)

        # Кнопки
        btn_frame = ttk.Frame(left_panel)
        btn_frame.pack(fill=tk.X, pady=5)

        ttk.Button(btn_frame, text='Рассчитать маршрут', command=self._calculate_route).pack(side=tk.LEFT, fill=tk.X,
                                                                                             expand=True, padx=2)
        ttk.Button(btn_frame, text='Показать все пути', command=self._show_all_paths).pack(side=tk.LEFT, fill=tk.X,
                                                                                           expand=True, padx=2)
        ttk.Button(btn_frame, text='Сбросить параметры', command=self._reset_parameters).pack(side=tk.LEFT, fill=tk.X,
                                                                                              expand=True, padx=2)

        # Правая панель
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Информация о маршруте
        info_frame = ttk.LabelFrame(right_panel, text='Информация о маршруте', padding=10)
        info_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(info_frame, text='Длина маршрута:').grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Label(info_frame, textvariable=self.distance_var).grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)

        ttk.Label(info_frame, text='Время доставки:').grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Label(info_frame, textvariable=self.time_var).grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)

        ttk.Label(info_frame, text='Пункты маршрута:').grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Label(info_frame, textvariable=self.route_var, wraplength=300).grid(row=2, column=1, sticky=tk.W, padx=5,
                                                                                pady=2)

        # Ограничения
        constraints_frame = ttk.LabelFrame(right_panel, text='Ограничения', padding=10)
        constraints_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(constraints_frame, text='Макс. нагрузка (кг):').grid(row=0, column=0, padx=5, pady=2)
        ttk.Entry(constraints_frame, textvariable=self.max_load_var, width=10).grid(row=0, column=1, padx=5, pady=2)

        ttk.Label(constraints_frame, text='Макс. время (ч):').grid(row=1, column=0, padx=5, pady=2)
        ttk.Entry(constraints_frame, textvariable=self.max_time_var, width=10).grid(row=1, column=1, padx=5, pady=2)

        ttk.Button(constraints_frame, text='Применить ограничения', command=self._apply_constraints).grid(
            row=2, column=0, columnspan=2, pady=5)

        # Результаты оптимизации
        results_frame = ttk.LabelFrame(right_panel, text='Результаты оптимизации', padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.results_text = tk.Text(results_frame, height=10, wrap=tk.WORD)
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        scrollbar = ttk.Scrollbar(results_frame, orient='vertical', command=self.results_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_text.config(yscrollcommand=scrollbar.set)

        # Кнопки экспорта
        export_frame = ttk.Frame(right_panel)
        export_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(export_frame, text='Экспорт маршрута (CSV)', command=self._export_route_csv).pack(side=tk.LEFT,
                                                                                                     padx=2)
        ttk.Button(export_frame, text='Экспорт маршрута (JSON)', command=self._export_route_json).pack(side=tk.LEFT,
                                                                                                       padx=2)
        ttk.Button(export_frame, text='Экспорт отчета', command=self._export_report).pack(side=tk.LEFT, padx=2)

    def _init_visualization_tab(self):
        """Инициализация вкладки визуализации"""
        control_frame = ttk.Frame(self.visualization_tab)
        control_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Button(control_frame, text='Обновить граф', command=self._update_graph).pack(side=tk.LEFT, padx=5)

        # Добавляем выбор типа карты
        map_type_frame = ttk.Frame(control_frame)
        map_type_frame.pack(side=tk.LEFT, padx=5)
        ttk.Label(map_type_frame, text='Тип карты:').pack(side=tk.LEFT)
        map_type_combo = ttk.Combobox(
            map_type_frame,
            textvariable=self.map_type_var,
            values=['OpenStreetMap', 'Satellite'],
            state='readonly',
            width=12
        )
        map_type_combo.pack(side=tk.LEFT, padx=5)

        ttk.Button(control_frame, text='Показать на карте', command=self._create_map_visualization).pack(side=tk.LEFT,
                                                                                                         padx=5)
        ttk.Button(control_frame, text='Сохранить изображение', command=self._save_graph_image).pack(side=tk.LEFT,
                                                                                                     padx=5)

    def _update_nodes_tree(self):
        """Обновляет таблицу узлов"""
        self.nodes_tree.delete(*self.nodes_tree.get_children())
        for node in self.node_manager.nodes:
            self.nodes_tree.insert('', 'end', values=(
                node['id'],
                node['name'],
                f"{node['latitude']:.6f}",
                f"{node['longitude']:.6f}"
            ))

    def _update_node_listbox(self):
        """Обновляет список узлов"""
        self.node_listbox.delete(0, tk.END)
        for node in sorted(self.node_manager.nodes, key=lambda x: x['id']):
            self.node_listbox.insert(tk.END, f"{node['id']}: {node['name']}")

    def _add_node_dialog(self):
        """Диалог добавления нового узла"""
        dialog = tk.Toplevel(self.root)
        dialog.title('Добавить пункт')
        dialog.geometry('300x250')
        dialog.resizable(False, False)
        dialog.grab_set()

        ttk.Label(dialog, text='Название пункта:').pack(pady=5)
        name_var = tk.StringVar()
        name_entry = ttk.Entry(dialog, textvariable=name_var)
        name_entry.pack(pady=5)
        name_entry.focus()

        ttk.Label(dialog, text='Широта (например: 55.751244):').pack(pady=5)
        lat_var = tk.StringVar()
        ttk.Entry(dialog, textvariable=lat_var).pack(pady=5)

        ttk.Label(dialog, text='Долгота (например: 37.618423):').pack(pady=5)
        lon_var = tk.StringVar()
        ttk.Entry(dialog, textvariable=lon_var).pack(pady=5)

        def save_node():
            name = name_var.get().strip()
            if not name:
                messagebox.showerror('Ошибка', 'Введите название пункта')
                return

            try:
                lat = float(lat_var.get())
                lon = float(lon_var.get())
                if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
                    raise ValueError('Некорректные координаты')

                self.node_manager.add_node(name, lat, lon)
                self._update_nodes_tree()
                self._update_node_listbox()
                self._calculate_edges()  # Автоматически пересчитываем связи
                dialog.destroy()
            except ValueError as e:
                messagebox.showerror('Ошибка', f'Некорректные данные: {str(e)}')

        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(pady=10)

        ttk.Button(btn_frame, text='Сохранить', command=save_node).pack(side=tk.LEFT, padx=10)
        ttk.Button(btn_frame, text='Отмена', command=dialog.destroy).pack(side=tk.RIGHT, padx=10)

    def _calculate_edges(self):
        """Пересчитывает связи между узлами"""
        try:
            avg_speed = float(self.avg_speed_var.get()) if self.avg_speed_var.get() else 50.0
            if avg_speed <= 0:
                raise ValueError('Скорость должна быть положительной')

            self.progress_label.config(text="Расчет маршрутов...")
            self.progress['value'] = 0
            self.root.update_idletasks()

            self.edge_manager.calculate_edges(avg_speed)
            self.graph_manager.update_graph()
            self._update_graph()

            self.progress['value'] = 100
            self.progress_label.config(text="Готово")

            if self.auto_calc_var.get() and len(self.node_listbox.curselection()) >= 2:
                self._calculate_route()
        except ValueError as e:
            messagebox.showerror('Ошибка', f'Некорректная скорость: {str(e)}')
            self.progress_label.config(text="Ошибка")
        except Exception as e:
            messagebox.showerror('Ошибка', f'Ошибка расчета: {str(e)}')
            self.progress_label.config(text="Ошибка")

    def _update_graph(self):
        """Обновляет визуализацию графа"""
        self.visualizer.update_visualization()
        self.canvas.draw()

    def _calculate_route(self):
        """Рассчитывает маршрут через выбранные узлы"""
        selected_indices = self.node_listbox.curselection()
        if len(selected_indices) < 2:
            messagebox.showwarning('Ошибка', 'Выберите хотя бы два пункта')
            return

        selected_nodes = []
        for index in selected_indices:
            node_str = self.node_listbox.get(index)
            node_id = int(node_str.split(":")[0])
            selected_nodes.append(node_id)

        try:
            # Создаем калькулятор маршрута
            calculator = RouteCalculatorFactory.create_calculator(
                self.algorithm_var.get(),
                self.graph_manager
            )

            # Создаем оптимизатор маршрутов
            optimizer = RouteOptimizer(
                self.node_manager,
                self.edge_manager,
                self.graph_manager,
                calculator
            )

            # Рассчитываем маршрут
            result = optimizer.calculate_route(selected_nodes)

            # Обновляем информацию о маршруте
            self.distance_var.set(f"{result['distance']:.2f} км")
            self.time_var.set(f"{result['time']:.2f} ч")
            self.route_var.set(' → '.join(result['route_names']))

            # Обновляем визуализацию
            self.visualizer.current_route = result['route']
            self._update_graph()

            # Обновляем результаты оптимизации
            self._update_optimization_results(result['distance'], result['time'], result['route_names'])

        except Exception as e:
            messagebox.showerror('Ошибка', f'Ошибка расчета: {str(e)}')

    def _show_all_paths(self):
        """Показывает все возможные пути"""
        if not self.node_manager.nodes or not self.edge_manager.edges:
            messagebox.showwarning('Ошибка', 'Добавьте пункты')
            return

        try:
            # Используем алгоритм Floyd-Warshall для всех путей
            all_pairs = nx.floyd_warshall(self.graph_manager.graph, weight='weight')

            result = 'Кратчайшие пути между всеми пунктами:\n\n'
            for i, source in enumerate(self.graph_manager.graph.nodes):
                for j, target in enumerate(self.graph_manager.graph.nodes):
                    if i < j:  # Чтобы избежать дублирования
                        source_name = next(
                            n['name'] for n in self.node_manager.nodes
                            if n['id'] == source
                        )
                        target_name = next(
                            n['name'] for n in self.node_manager.nodes
                            if n['id'] == target
                        )
                        distance = all_pairs[source][target]
                        result += f"{source_name} → {target_name}: {distance:.2f} км\n"

            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, result)
            self.results_text.see(tk.END)

        except nx.NetworkXError as e:
            messagebox.showerror('Ошибка', f'Ошибка расчета путей: {str(e)}')

    def _apply_constraints(self):
        """Применяет ограничения к маршруту"""
        constraints = []

        if self.max_load_var.get():
            try:
                max_load = float(self.max_load_var.get())
                if max_load <= 0:
                    raise ValueError('Нагрузка должна быть положительной')
                constraints.append(f'Максимальная нагрузка: {max_load} кг')
            except ValueError:
                messagebox.showerror('Ошибка', 'Некорректное значение нагрузки')

        if self.max_time_var.get():
            try:
                max_time = float(self.max_time_var.get())
                if max_time <= 0:
                    raise ValueError('Время должно быть положительным')
                constraints.append(f'Максимальное время: {max_time} ч')
            except ValueError:
                messagebox.showerror('Ошибка', 'Некорректное значение времени')

        if constraints:
            self.results_text.insert(tk.END, '\n\nПрименены ограничения:\n' + '\n'.join(constraints))
            self.results_text.see(tk.END)

    def _update_optimization_results(self, distance: float, total_time: float, route_names: List[str]):
        """Обновляет результаты оптимизации"""
        result = f'Оптимальный маршрут: {" → ".join(route_names)}\n'
        result += f'Общее расстояние: {distance:.2f} км\n'
        result += f'Общее время: {total_time:.2f} ч\n'

        # Расчет стоимости доставки (примерная формула)
        cost_per_km = 15  # руб/км
        cost = distance * cost_per_km
        result += f'Примерная стоимость: {cost:.2f} руб\n'

        # Проверка ограничений
        if self.max_time_var.get():
            try:
                max_time = float(self.max_time_var.get())
                if total_time > max_time:
                    result += f'\n⚠️ Время доставки ({total_time:.2f} ч) превышает максимальное ({max_time} ч)!'
            except:
                pass

        if self.max_load_var.get():
            try:
                max_load = float(self.max_load_var.get())
                result += f'\nМаксимальная нагрузка: {max_load} кг (проверьте соответствие)'
            except:
                pass

        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, result)
        self.results_text.see(tk.END)

    def _create_map_visualization(self):
        """Создает визуализацию маршрута на карте с использованием реальных дорог"""
        if not hasattr(self.visualizer, 'current_route') or not self.visualizer.current_route:
            messagebox.showwarning('Ошибка', 'Нет маршрута для визуализации')
            return

        try:
            # Устанавливаем тип карты в визуализаторе
            self.visualizer.map_type = self.map_type_var.get()

            route_info = {
                'route_names': [
                    self.node_manager.get_node_by_id(node_id)['name']
                    for node_id in self.visualizer.current_route
                    if self.node_manager.get_node_by_id(node_id)
                ]
            }

            m = self.visualizer.create_map_visualization(
                self.visualizer.current_route,
                route_info
            )

            # Сохраняем карту во временный файл и открываем в браузере
            map_file = os.path.join(os.getenv('TEMP', '.'), 'temp_route_map.html')
            m.save(map_file)

            # Удаляем предыдущий временный файл, если он существует
            if hasattr(self, 'last_map_file') and self.last_map_file and os.path.exists(self.last_map_file):
                try:
                    os.remove(self.last_map_file)
                except:
                    pass

            webbrowser.open(f'file://{os.path.abspath(map_file).replace(os.sep, "/")}')
            self.last_map_file = map_file

        except Exception as e:
            messagebox.showerror('Ошибка', f'Не удалось создать карту: {str(e)}')

    def _load_csv(self):
        """Загружает данные из CSV файла"""
        file_path = filedialog.askopenfilename(
            title='Выберите CSV файл',
            filetypes=[('CSV files', '*.csv'), ('All files', '*.*')]
        )
        if not file_path:
            return

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                self.node_manager.clear_nodes()
                nodes_added = 0

                for row in reader:
                    if 'latitude' in row and 'longitude' in row:  # Это узел
                        try:
                            self.node_manager.add_node(
                                row.get('name', ''),
                                float(row['latitude']),
                                float(row['longitude'])
                            )
                            nodes_added += 1
                        except (ValueError, KeyError) as e:
                            print(f"Ошибка обработки строки: {str(e)}")

                if nodes_added > 0:
                    self._update_nodes_tree()
                    self._update_node_listbox()
                    self._calculate_edges()
                    messagebox.showinfo('Успех', f'Успешно загружено {nodes_added} пунктов из CSV')
                else:
                    messagebox.showwarning('Предупреждение', 'Не найдено подходящих данных в CSV файле')

        except Exception as e:
            messagebox.showerror('Ошибка', f'Ошибка загрузки CSV: {str(e)}')

    def _load_json(self):
        """Загружает данные из JSON файла"""
        file_path = filedialog.askopenfilename(
            title='Выберите JSON файл',
            filetypes=[('JSON files', '*.json'), ('All files', '*.*')]
        )
        if not file_path:
            return

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

                if 'nodes' not in data:
                    raise ValueError('Некорректный формат JSON файла')

                self.node_manager.clear_nodes()
                for node in data['nodes']:
                    try:
                        self.node_manager.add_node(
                            node.get('name', ''),
                            node['latitude'],
                            node['longitude']
                        )
                    except (ValueError, KeyError) as e:
                        print(f"Ошибка обработки узла: {str(e)}")

                self._update_nodes_tree()
                self._update_node_listbox()

                # Загружаем ребра, если они есть в файле
                if 'edges' in data:
                    self.edge_manager.edges = data['edges']
                    self.graph_manager.update_graph()
                else:
                    self._calculate_edges()

                messagebox.showinfo('Успех', f'Успешно загружено {len(data["nodes"])} пунктов из JSON')

        except Exception as e:
            messagebox.showerror('Ошибка', f'Ошибка загрузки JSON: {str(e)}')

    def _save_csv(self):
        """Сохраняет данные в CSV файл"""
        file_path = filedialog.asksaveasfilename(
            title='Сохранить как CSV',
            defaultextension='.csv',
            filetypes=[('CSV files', '*.csv'), ('All files', '*.*')]
        )
        if not file_path:
            return

        try:
            # Подготавливаем данные для экспорта
            data = []
            data.append(['id', 'name', 'latitude', 'longitude'])
            for node in self.node_manager.nodes:
                data.append([node['id'], node['name'], node['latitude'], node['longitude']])

            data.append([])
            data.append(['from', 'to', 'distance', 'time'])
            for edge in self.edge_manager.edges:
                data.append([edge['from'], edge['to'], edge['distance'], edge['time']])

            # Экспортируем
            exporter = CSVExporter()
            exporter.export(data, file_path)
            messagebox.showinfo('Успех', 'Данные успешно сохранены в CSV')

        except Exception as e:
            messagebox.showerror('Ошибка', f'Ошибка сохранения CSV: {str(e)}')

    def _save_json(self):
        """Сохраняет данные в JSON файл"""
        file_path = filedialog.asksaveasfilename(
            title='Сохранить как JSON',
            defaultextension='.json',
            filetypes=[('JSON files', '*.json'), ('All files', '*.*')]
        )
        if not file_path:
            return

        data = {
            'nodes': self.node_manager.get_nodes_data(),
            'edges': self.edge_manager.edges
        }

        try:
            exporter = JSONExporter()
            exporter.export(data, file_path)
            messagebox.showinfo('Успех', 'Данные успешно сохранены в JSON')
        except Exception as e:
            messagebox.showerror('Ошибка', f'Ошибка сохранения JSON: {str(e)}')

    def _export_route_csv(self):
        """Экспортирует текущий маршрут в CSV"""
        if not hasattr(self.visualizer, 'current_route') or not self.visualizer.current_route:
            messagebox.showwarning('Ошибка', 'Нет текущего маршрута для экспорта')
            return

        file_path = filedialog.asksaveasfilename(
            title='Экспорт маршрута как CSV',
            defaultextension='.csv',
            filetypes=[('CSV files', '*.csv'), ('All files', '*.*')]
        )
        if not file_path:
            return

        try:
            # Подготавливаем данные для экспорта
            data = []
            data.append(['Маршрут', self.route_var.get()])
            data.append(['Расстояние (км)', self.distance_var.get()])
            data.append(['Время (ч)', self.time_var.get()])
            data.append(['Алгоритм', self.algorithm_var.get()])
            data.append([])
            data.append(['Порядок', 'ID', 'Название', 'Широта', 'Долгота'])

            for i, node_id in enumerate(self.visualizer.current_route, 1):
                node = self.node_manager.get_node_by_id(node_id)
                if node:
                    data.append([
                        i,
                        node['id'],
                        node['name'],
                        node['latitude'],
                        node['longitude']
                    ])

            # Экспортируем
            exporter = CSVExporter()
            exporter.export(data, file_path)
            messagebox.showinfo('Успех', 'Маршрут успешно экспортирован в CSV')

        except Exception as e:
            messagebox.showerror('Ошибка', f'Ошибка экспорта: {str(e)}')

    def _export_route_json(self):
        """Экспортирует текущий маршрут в JSON"""
        if not hasattr(self.visualizer, 'current_route') or not self.visualizer.current_route:
            messagebox.showwarning('Ошибка', 'Нет текущего маршрута для экспорта')
            return

        file_path = filedialog.asksaveasfilename(
            title='Экспорт маршрута как JSON',
            defaultextension='.json',
            filetypes=[('JSON files', '*.json'), ('All files', '*.*')]
        )
        if not file_path:
            return

        route_nodes = []
        for node_id in self.visualizer.current_route:
            node = self.node_manager.get_node_by_id(node_id)
            if node:
                route_nodes.append({
                    'id': node['id'],
                    'name': node['name'],
                    'latitude': node['latitude'],
                    'longitude': node['longitude']
                })

        route_data = {
            'path': self.visualizer.current_route,
            'distance': self.distance_var.get(),
            'time': self.time_var.get(),
            'nodes': route_nodes,
            'algorithm': self.algorithm_var.get()
        }

        try:
            exporter = JSONExporter()
            exporter.export(route_data, file_path)
            messagebox.showinfo('Успех', 'Маршрут успешно экспортирован в JSON')
        except Exception as e:
            messagebox.showerror('Ошибка', f'Ошибка экспорта: {str(e)}')

    def _export_report(self):
        """Экспортирует отчет в текстовый файл"""
        file_path = filedialog.asksaveasfilename(
            title='Экспорт отчета',
            defaultextension='.txt',
            filetypes=[('Text files', '*.txt'), ('All files', '*.*')]
        )
        if not file_path:
            return

        report = self.results_text.get(1.0, tk.END)

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(report)
            messagebox.showinfo('Успех', 'Отчет успешно экспортирован')
        except Exception as e:
            messagebox.showerror('Ошибка', f'Ошибка экспорта: {str(e)}')

    def _save_graph_image(self):
        """Сохраняет изображение графа"""
        file_path = filedialog.asksaveasfilename(
            title='Сохранить изображение',
            defaultextension='.png',
            filetypes=[('PNG files', '*.png'), ('All files', '*.*')]
        )
        if not file_path:
            return

        try:
            self.visualizer.fig.savefig(file_path, dpi=300, bbox_inches='tight',
                                        facecolor=self.visualizer.fig.get_facecolor())
            messagebox.showinfo('Успех', 'Изображение сохранено')
        except Exception as e:
            messagebox.showerror('Ошибка', f'Ошибка сохранения: {str(e)}')

    def _clear_all(self):
        """Очищает все данные"""
        if messagebox.askyesno('Подтверждение', 'Вы уверены, что хотите очистить все данные?'):
            self.node_manager.clear_nodes()
            self.edge_manager.clear_edges()
            self.graph_manager.graph.clear()
            self.visualizer.current_route = []
            self._update_nodes_tree()
            self._update_node_listbox()
            self.visualizer.ax.clear()
            self.canvas.draw()
            self.results_text.delete(1.0, tk.END)
            self.distance_var.set('0 км')
            self.time_var.set('0 ч')
            self.route_var.set('')
            self.progress['value'] = 0
            self.progress_label.config(text="Готово")

    def _reset_parameters(self):
        """Сбрасывает параметры расчета"""
        self.node_listbox.selection_clear(0, tk.END)
        self.algorithm_var.set('dijkstra')
        self.avg_speed_var.set('50')
        self.auto_calc_var.set(True)
        self.max_load_var.set('')
        self.max_time_var.set('')
        self.distance_var.set('0 км')
        self.time_var.set('0 ч')
        self.route_var.set('')
        self.results_text.delete(1.0, tk.END)
        self.visualizer.current_route = []
        self._update_graph()


if __name__ == '__main__':
    root = tk.Tk()
    try:
        app = DeliveryOptimizerApp(root)
        root.mainloop()
    except Exception as e:
        messagebox.showerror("Critical Error", f"Application crashed: {str(e)}")
        raise