import { MapContainer, TileLayer, CircleMarker, Polyline, Tooltip } from 'react-leaflet';
import type { MatchingScenario, MatchingTraderStats } from '../../api/types';

interface Props {
  scenario: MatchingScenario | null;
  chromosome: number[] | null;
  traderStats: MatchingTraderStats[] | null;
}

export default function MatchingMap({ scenario, chromosome, traderStats }: Props) {
  if (!scenario) {
    return <div className="map-empty">Carregue um cenário pra começar</div>;
  }
  const center: [number, number] = [-18, -52];

  const overloadSet = new Set<number>(
    (traderStats ?? [])
      .filter(s => s.overCapacity)
      .map(s => s.traderId)
  );

  return (
    <MapContainer center={center} zoom={5} scrollWheelZoom={true} style={{ height: '100%', width: '100%' }}>
      <TileLayer
        attribution='&copy; OpenStreetMap'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />

      {/* Porto Santos */}
      <CircleMarker
        center={[scenario.portLat, scenario.portLng]}
        radius={10}
        pathOptions={{ color: '#000', fillColor: '#444', fillOpacity: 0.9 }}
      >
        <Tooltip direction="top">Porto de Santos</Tooltip>
      </CircleMarker>

      {/* Trader hubs */}
      {scenario.traders.map(t => (
        <CircleMarker
          key={`trader-${t.id}`}
          center={[t.hubLat, t.hubLng]}
          radius={9}
          pathOptions={{ color: t.cor, fillColor: t.cor, fillOpacity: 0.95, weight: 2 }}
        >
          <Tooltip direction="top">
            <strong>{t.nome}</strong> — {t.hubMunicipio}<br />
            cap: {t.capacidadeT.toFixed(0)} t · prot ≥ {t.proteinaMin}
          </Tooltip>
        </CircleMarker>
      ))}

      {/* Produtores */}
      {scenario.producers.map(p => {
        const lotIdx = scenario.lots.findIndex(l => l.producerId === p.id);
        const matched = chromosome && lotIdx >= 0 ? chromosome[lotIdx] : -1;
        const cor = matched >= 0 ? scenario.traders[matched].cor : '#aaa';
        return (
          <CircleMarker
            key={`prod-${p.id}`}
            center={[p.lat, p.lng]}
            radius={6}
            pathOptions={{ color: cor, fillColor: cor, fillOpacity: 0.85 }}
          >
            <Tooltip direction="top">
              <strong>{p.nome}</strong> — {p.municipio}/{p.uf}
              {lotIdx >= 0 && (
                <>
                  <br />vol: {scenario.lots[lotIdx].volumeT.toFixed(0)} t · prot: {scenario.lots[lotIdx].proteina.toFixed(1)}
                </>
              )}
            </Tooltip>
          </CircleMarker>
        );
      })}

      {/* Linhas produtor→trader (matched) */}
      {chromosome && scenario.producers.map(p => {
        const lotIdx = scenario.lots.findIndex(l => l.producerId === p.id);
        if (lotIdx < 0) return null;
        const j = chromosome[lotIdx];
        if (j < 0 || j >= scenario.traders.length) return null;
        const trader = scenario.traders[j];
        return (
          <Polyline
            key={`pt-${p.id}`}
            positions={[[p.lat, p.lng], [trader.hubLat, trader.hubLng]]}
            pathOptions={{ color: trader.cor, weight: 2, opacity: 0.8 }}
          />
        );
      })}

      {/* Linhas trader→porto (todos os traders, grosso) */}
      {scenario.traders.map(t => {
        const overload = overloadSet.has(t.id);
        return (
          <Polyline
            key={`tp-${t.id}`}
            positions={[[t.hubLat, t.hubLng], [scenario.portLat, scenario.portLng]]}
            pathOptions={{
              color: overload ? '#e63946' : t.cor,
              weight: 4,
              opacity: 0.85,
              dashArray: overload ? '8 6' : undefined,
            }}
          />
        );
      })}
    </MapContainer>
  );
}
