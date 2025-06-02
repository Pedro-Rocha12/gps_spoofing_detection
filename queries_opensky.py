def query_flights_between(
    origin: str,
    destination: str,
    start_ts: int,
    end_ts: int,
    limit: int = 10000
) -> str:
    """
    Retorna SQL para buscar voos entre origin e destination no intervalo [start_ts, end_ts).
    origin/destination: ICAO dos aeroportos (e.g. 'SBSP', 'SBBR').
    start_ts, end_ts: unix timestamps para início/fim de temporada.
    """
    return f"""
    SELECT
        icao24, callsign,
        estdepartureairport, estarrivalairport,
        firstseen, lastseen,
        takeofftime, landingtime,
        takeofflatitude, takeofflongitude,
        landinglatitude, landinglongitude
    FROM minio.osky.flights_data5
    WHERE estdepartureairport = '{origin}'
      AND estarrivalairport   = '{destination}'
      AND firstseen >= {start_ts}
      AND firstseen <  {end_ts}
      AND callsign IS NOT NULL
    ORDER BY firstseen
    LIMIT {limit}
    """

def query_unpacked_track(
    origin: str,
    destination: str,
    start_ts: int,
    end_ts: int,
    limit_flights: int = 1000
) -> str:
    """
    Desempacota o array `track` de até N voos dessa rota/intervalo.
    """
    return f"""
    WITH sel AS (
      SELECT icao24, callsign, track
      FROM minio.osky.flights_data5
      WHERE estdepartureairport = '{origin}'
        AND estarrivalairport   = '{destination}'
        AND firstseen >= {start_ts}
        AND firstseen <  {end_ts}
        AND callsign IS NOT NULL
      ORDER BY firstseen
      LIMIT {limit_flights}
    )
    SELECT
      s.icao24, s.callsign,
      t.time    AS time,
      t.latitude    AS lat,
      t.longitude   AS lon,
      t.altitude    AS baroaltitude,
      t.heading     AS heading,
      t.onground    AS onground
    FROM sel s
    CROSS JOIN UNNEST(s.track) AS t (
      time, latitude, longitude, altitude, heading, onground
    )
    ORDER BY s.icao24, t.time
    """

def query_detalhes_voo(
    icao24: str,
    callsign: str,
    dep: str,
    arr: str,
    firstseen: int,
    lastseen: int
) -> str:
    """
    Busca todos os campos de flights_data5 para um voo específico.
    """
    return f"""
    SELECT *
    FROM minio.osky.flights_data5
    WHERE icao24 = '{icao24}'
      AND TRIM(callsign) = '{callsign}'
      AND estdepartureairport = '{dep}'
      AND estarrivalairport   = '{arr}'
      AND firstseen = {firstseen}
      AND lastseen  = {lastseen}
    """

def query_state_vectors_for_flight(
    icao24: str,
    firstseen: int,
    lastseen: int
) -> str:
    """
    Busca os vetores de estado (state_vectors_data4) para um voo específico.
    """
    return f"""
    SELECT
        time, icao24, lat, lon,
        velocity, heading, vertrate,
        baroaltitude, geoaltitude, callsign
    FROM minio.osky.state_vectors_data4
    WHERE icao24 = '{icao24}'
      AND time BETWEEN {firstseen} AND {lastseen}
      AND lat IS NOT NULL
      AND lon IS NOT NULL
    ORDER BY time
    """

def query_track_for_flight(
    icao24: str,
    callsign: str,
    dep: str,
    arr: str,
    firstseen: int,
    lastseen: int
) -> str:
    """
    Desempacota o array `track` de flights_data5 para um voo específico.
    """
    return f"""
    WITH sel AS (
      SELECT track
      FROM minio.osky.flights_data5
      WHERE icao24 = '{icao24}'
        AND callsign = '{callsign}'
        AND estdepartureairport = '{dep}'
        AND estarrivalairport   = '{arr}'
        AND firstseen = {firstseen}
        AND lastseen  = {lastseen}
    )
    SELECT
      t.time         AS time,
      t.latitude     AS lat,
      t.longitude    AS lon,
      t.altitude     AS baroaltitude,
      t.heading      AS heading,
      t.onground     AS onground
    FROM sel
    CROSS JOIN UNNEST(sel.track) AS t (
      time, latitude, longitude, altitude, heading, onground
    )
    ORDER BY t.time
    """

def query_list_flights_by_icao(
    icao24: str,
    start_ts: int,
    end_ts: int,
    limit: int = 10
) -> str:
    """
    Retorna os metadados (icao24, callsign, firstseen, lastseen, dep, arr)
    dos N primeiros voos para um dado icao24 nesse período.
    """
    return f"""
    SELECT
      icao24,
      callsign,
      estdepartureairport,
      estarrivalairport,
      firstseen,
      lastseen
    FROM minio.osky.flights_data5
    WHERE icao24 = '{icao24}'
      AND firstseen >= {start_ts}
      AND firstseen <  {end_ts}
    ORDER BY firstseen
    LIMIT {limit}
    """