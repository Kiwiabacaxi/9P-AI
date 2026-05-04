package tsp

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// =============================================================================
// Cliente OSRM (Open Source Routing Machine)
//
// Usamos o servidor demo público (router.project-osrm.org), que é gratuito,
// não exige API key e usa dados do OpenStreetMap. Endpoints usados:
//
//   table/v1/driving/{c1};{c2};...   → matriz N×N de distâncias por estrada
//   route/v1/driving/{c1};{c2};...   → polyline (geometry) de uma rota completa
//
// Coordenadas em todos os endpoints OSRM seguem o padrão GeoJSON: [lng, lat]
// (oposto do Leaflet, que é [lat, lng]). Cuidado nas conversões.
//
// ⚠️ Limites informais do servidor demo: ~1 req/s, máx 100 coordenadas por
// requisição. Aplicamos cache em memória pra reusar matrizes/geometrias e
// evitar tráfego desnecessário. Cache invalida quando o conjunto de cidades muda.
// =============================================================================

const (
	osrmBaseURL    = "https://router.project-osrm.org"
	osrmHTTPClient = 12 // segundos de timeout
)

var osrmHTTP = &http.Client{Timeout: osrmHTTPClient * time.Second}

// formatCoords — junta as cidades em "lng,lat;lng,lat;..." na ordem dada.
// Se ordem == nil, usa todas as cidades em ordem natural.
func formatCoords(cidades []Cidade, ordem []int) string {
	if ordem == nil {
		ordem = make([]int, len(cidades))
		for i := range ordem {
			ordem[i] = i
		}
	}
	parts := make([]string, len(ordem))
	for i, idx := range ordem {
		c := cidades[idx]
		// 6 casas decimais já é precisão de ~10cm, suficiente.
		parts[i] = fmt.Sprintf("%.6f,%.6f", c.Lng, c.Lat)
	}
	return strings.Join(parts, ";")
}

// =============================================================================
// /table — matriz N×N de distâncias.
// =============================================================================

type osrmTableResp struct {
	Code      string      `json:"code"`
	Message   string      `json:"message,omitempty"`
	Distances [][]float64 `json:"distances"` // metros
	Durations [][]float64 `json:"durations"` // segundos
}

// FetchOSRMMatrix — matriz N×N em km. Faz uma requisição /table só.
func FetchOSRMMatrix(cidades []Cidade) ([][]float64, error) {
	if len(cidades) < 2 {
		return nil, errors.New("matriz OSRM precisa de >= 2 cidades")
	}
	if len(cidades) > 100 {
		return nil, errors.New("OSRM demo aceita no máximo 100 cidades por requisição")
	}
	url := fmt.Sprintf("%s/table/v1/driving/%s?annotations=distance",
		osrmBaseURL, formatCoords(cidades, nil))

	resp, err := osrmHTTP.Get(url)
	if err != nil {
		return nil, fmt.Errorf("OSRM table: %w", err)
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("OSRM table HTTP %d: %s", resp.StatusCode, string(body))
	}
	var data osrmTableResp
	if err := json.Unmarshal(body, &data); err != nil {
		return nil, fmt.Errorf("OSRM table parse: %w", err)
	}
	if data.Code != "Ok" {
		return nil, fmt.Errorf("OSRM table erro: %s — %s", data.Code, data.Message)
	}
	if len(data.Distances) != len(cidades) {
		return nil, fmt.Errorf("OSRM table tamanho inesperado: %d (esperado %d)", len(data.Distances), len(cidades))
	}
	// metros → km
	n := len(data.Distances)
	matriz := make([][]float64, n)
	for i := range matriz {
		matriz[i] = make([]float64, n)
		for j := 0; j < n; j++ {
			matriz[i][j] = data.Distances[i][j] / 1000.0
		}
	}
	return matriz, nil
}

// =============================================================================
// /route — polyline real do tour pelas estradas.
// =============================================================================

type osrmRouteResp struct {
	Code    string `json:"code"`
	Message string `json:"message,omitempty"`
	Routes  []struct {
		Distance float64 `json:"distance"` // metros
		Duration float64 `json:"duration"` // segundos
		Geometry struct {
			Type        string      `json:"type"`
			Coordinates [][]float64 `json:"coordinates"` // [lng, lat]
		} `json:"geometry"`
		Legs []struct {
			Distance float64 `json:"distance"` // metros
			Duration float64 `json:"duration"` // segundos
			Steps    []struct {
				Geometry struct {
					Coordinates [][]float64 `json:"coordinates"` // [lng, lat]
				} `json:"geometry"`
			} `json:"steps"`
		} `json:"legs"`
	} `json:"routes"`
}

// LegGeometry — geometria curvada de UM leg (entre dois cidades consecutivas no tour).
// Permite que o frontend anime o truck seguindo as curvas reais das estradas
// dentro de cada perna, em vez de cortar reto entre as cidades.
type LegGeometry struct {
	Polyline  [][2]float64 `json:"polyline"`  // [[lat, lng], ...] da perna
	Distancia float64      `json:"distancia"` // km
	Duracao   float64      `json:"duracao"`   // segundos
	DeID      int          `json:"deId"`      // city id no início da perna
	ParaID    int          `json:"paraId"`    // city id no fim da perna
}

// RouteGeometry — polyline percorrendo as cidades na ordem do tour, fechando
// o ciclo (volta a c0). Retorna pontos em [lat, lng] (formato Leaflet) e
// distância total em km.
type RouteGeometry struct {
	Polyline  [][2]float64  `json:"polyline"`  // [[lat, lng], ...] tour fechado completo
	Legs      []LegGeometry `json:"legs"`      // geometria por perna (entre cidades consecutivas)
	Distancia float64       `json:"distancia"` // km totais
	Duracao   float64       `json:"duracao"`   // segundos totais
}

// FetchOSRMRoute — geometria curvada da rota completa pelo tour fechado.
// Usa steps=true pra obter geometria por step e poder consolidar legs separados.
func FetchOSRMRoute(cidades []Cidade, tour []int) (*RouteGeometry, error) {
	if len(tour) < 2 {
		return nil, errors.New("rota OSRM precisa de >= 2 cidades")
	}
	// monta a sequência: tour + retorno ao primeiro
	ordem := make([]int, 0, len(tour)+1)
	ordem = append(ordem, tour...)
	ordem = append(ordem, tour[0])

	url := fmt.Sprintf("%s/route/v1/driving/%s?overview=full&geometries=geojson&steps=true",
		osrmBaseURL, formatCoords(cidades, ordem))

	resp, err := osrmHTTP.Get(url)
	if err != nil {
		return nil, fmt.Errorf("OSRM route: %w", err)
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("OSRM route HTTP %d: %s", resp.StatusCode, string(body))
	}
	var data osrmRouteResp
	if err := json.Unmarshal(body, &data); err != nil {
		return nil, fmt.Errorf("OSRM route parse: %w", err)
	}
	if data.Code != "Ok" || len(data.Routes) == 0 {
		return nil, fmt.Errorf("OSRM route erro: %s — %s", data.Code, data.Message)
	}

	route := data.Routes[0]

	// polyline geral (overview=full)
	geom := route.Geometry.Coordinates
	out := make([][2]float64, len(geom))
	for i, c := range geom {
		// GeoJSON [lng, lat] → Leaflet [lat, lng]
		out[i] = [2]float64{c[1], c[0]}
	}

	// legs: concatena steps[].geometry pra obter polyline de cada perna
	legs := make([]LegGeometry, 0, len(route.Legs))
	for li, leg := range route.Legs {
		// Os legs do OSRM correspondem 1:1 com os pares (ordem[li], ordem[li+1]),
		// que por sua vez vêm de tour[li] → tour[(li+1) % len(tour)] (último é volta).
		deID := tour[li%len(tour)]
		paraID := tour[(li+1)%len(tour)]

		// concatena geometrias de todos os steps; o primeiro ponto do step k+1
		// é o último do step k (joining), então pulamos duplicatas.
		var pts [][2]float64
		for si, step := range leg.Steps {
			coords := step.Geometry.Coordinates
			startIdx := 0
			if si > 0 && len(coords) > 0 {
				startIdx = 1 // pula ponto duplicado na junção
			}
			for _, c := range coords[startIdx:] {
				pts = append(pts, [2]float64{c[1], c[0]})
			}
		}
		// Fallback: se o OSRM não retornou steps por algum motivo, usa um
		// segmento reto entre as cidades — animação degrada mas não quebra.
		if len(pts) < 2 {
			a := cidades[deID]
			b := cidades[paraID]
			pts = [][2]float64{{a.Lat, a.Lng}, {b.Lat, b.Lng}}
		}
		legs = append(legs, LegGeometry{
			Polyline:  pts,
			Distancia: leg.Distance / 1000.0,
			Duracao:   leg.Duration,
			DeID:      deID,
			ParaID:    paraID,
		})
	}

	return &RouteGeometry{
		Polyline:  out,
		Legs:      legs,
		Distancia: route.Distance / 1000.0,
		Duracao:   route.Duration,
	}, nil
}
