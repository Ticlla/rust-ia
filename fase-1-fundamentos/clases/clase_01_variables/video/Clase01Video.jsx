import React from 'react';
import {
  AbsoluteFill,
  interpolate,
  Sequence,
  spring,
  useCurrentFrame,
  useVideoConfig,
} from 'remotion';

// Componente para ventana de código estilo macOS
const CodeCard = ({ title, children, borderColor = '#334155' }) => (
  <div
    style={{
      backgroundColor: '#0f172a',
      borderRadius: 24,
      border: `2px solid ${borderColor}`,
      boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.7)',
      overflow: 'hidden',
      width: '100%',
      maxWidth: 920,
    }}
  >
    {/* Barra superior de ventana */}
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        padding: '16px 24px',
        backgroundColor: '#1e293b',
        borderBottom: '1px solid #334155',
      }}
    >
      <div style={{ display: 'flex', gap: 10 }}>
        <div style={{ width: 14, height: 14, borderRadius: '50%', backgroundColor: '#ef4444' }} />
        <div style={{ width: 14, height: 14, borderRadius: '50%', backgroundColor: '#f59e0b' }} />
        <div style={{ width: 14, height: 14, borderRadius: '50%', backgroundColor: '#10b981' }} />
      </div>
      <span style={{ marginLeft: 20, fontSize: 20, color: '#94a3b8', fontFamily: 'monospace' }}>
        {title}
      </span>
    </div>
    {/* Contenido de código */}
    <div style={{ padding: 32, fontFamily: 'monospace', fontSize: 28, lineHeight: 1.6 }}>
      {children}
    </div>
  </div>
);

// Escena 1: Introducción
const SceneIntro = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const titleScale = spring({ frame, fps, config: { damping: 12 } });
  const badgeOpacity = interpolate(frame, [10, 25], [0, 1], { extrapolateRight: 'clamp' });
  const descOpacity = interpolate(frame, [25, 40], [0, 1], { extrapolateRight: 'clamp' });

  return (
    <AbsoluteFill
      style={{
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        alignItems: 'center',
        padding: 60,
        textAlign: 'center',
      }}
    >
      <div
        style={{
          opacity: badgeOpacity,
          padding: '12px 28px',
          borderRadius: 9999,
          background: 'rgba(249, 115, 22, 0.2)',
          border: '2px solid #f97316',
          color: '#fb923c',
          fontSize: 26,
          fontWeight: 'bold',
          letterSpacing: 2,
          marginBottom: 30,
        }}
      >
        🦀 RUST PARA IA • FASE 01
      </div>

      <h1
        style={{
          transform: `scale(${titleScale})`,
          fontSize: 68,
          fontWeight: 900,
          color: '#ffffff',
          lineHeight: 1.15,
          marginBottom: 24,
        }}
      >
        ¿Por qué Rust{' '}
        <span
          style={{
            background: 'linear-gradient(to right, #f97316, #ef4444)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
          }}
        >
          no te deja cambiar
        </span>{' '}
        tus variables?
      </h1>

      <p
        style={{
          opacity: descOpacity,
          fontSize: 32,
          color: '#94a3b8',
          maxWidth: 800,
        }}
      >
        Clase 01: Inmutabilidad por Defecto y Seguridad de Memoria en Pipelines de Entrenamiento
      </p>
    </AbsoluteFill>
  );
};

// Escena 2: Inmutabilidad por defecto
const SceneInmutabilidad = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const cardScale = spring({ frame, fps, config: { damping: 14 } });
  const errorOpacity = interpolate(frame, [25, 40], [0, 1], { extrapolateRight: 'clamp' });
  const errorSlide = interpolate(frame, [25, 40], [30, 0], { extrapolateRight: 'clamp' });

  return (
    <AbsoluteFill
      style={{
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        alignItems: 'center',
        padding: 60,
      }}
    >
      <div
        style={{
          color: '#38bdf8',
          fontSize: 26,
          fontWeight: 'bold',
          marginBottom: 20,
          letterSpacing: 1.5,
        }}
      >
        1. INMUTABILIDAD POR DEFECTO
      </div>

      <div style={{ transform: `scale(${cardScale})`, width: '100%', maxWidth: 920 }}>
        <CodeCard title="main.rs (Rust Compiler)" borderColor="#ef4444">
          <div style={{ color: '#94a3b8' }}>// Rust asume que TODO es inmutable:</div>
          <div>
            <span style={{ color: '#f43f5e' }}>let </span>
            <span style={{ color: '#f8fafc' }}>learning_rate</span>
            <span style={{ color: '#94a3b8' }}>: </span>
            <span style={{ color: '#38bdf8' }}>f32 </span>
            <span style={{ color: '#94a3b8' }}>= </span>
            <span style={{ color: '#34d399' }}>0.01</span>
            <span style={{ color: '#94a3b8' }}>;</span>
          </div>
          <div style={{ marginTop: 12 }}>
            <span style={{ color: '#f8fafc' }}>learning_rate </span>
            <span style={{ color: '#94a3b8' }}>= </span>
            <span style={{ color: '#34d399' }}>0.005</span>
            <span style={{ color: '#94a3b8' }}>;</span>
          </div>

          <div
            style={{
              opacity: errorOpacity,
              transform: `translateY(${errorSlide}px)`,
              marginTop: 24,
              padding: 16,
              backgroundColor: 'rgba(239, 68, 68, 0.15)',
              borderLeft: '4px solid #ef4444',
              borderRadius: 8,
              color: '#fca5a5',
              fontSize: 22,
            }}
          >
            ❌ <strong>error[E0384]:</strong> cannot assign twice to immutable variable `learning_rate`
          </div>
        </CodeCard>
      </div>

      <div
        style={{
          opacity: errorOpacity,
          marginTop: 35,
          textAlign: 'center',
          color: '#cbd5e1',
          fontSize: 28,
          maxWidth: 820,
        }}
      >
        🛡️ <strong>Ventaja en IA:</strong> Si 8 GPUs leen tus hiperparámetros en paralelo, ninguna puede alterarlos en secreto. <em>Cero Data Races.</em>
      </div>
    </AbsoluteFill>
  );
};

// Escena 3: Mutabilidad Explícita
const SceneMutabilidad = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const cardScale = spring({ frame, fps, config: { damping: 14 } });
  const checkOpacity = interpolate(frame, [20, 35], [0, 1], { extrapolateRight: 'clamp' });

  return (
    <AbsoluteFill
      style={{
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        alignItems: 'center',
        padding: 60,
      }}
    >
      <div
        style={{
          color: '#34d399',
          fontSize: 26,
          fontWeight: 'bold',
          marginBottom: 20,
          letterSpacing: 1.5,
        }}
      >
        2. MUTABILIDAD EXPLÍCITA: `let mut`
      </div>

      <div style={{ transform: `scale(${cardScale})`, width: '100%', maxWidth: 920 }}>
        <CodeCard title="training_loop.rs" borderColor="#10b981">
          <div style={{ color: '#94a3b8' }}>// Si un valor debe cambiar, debes ser EXPLÍCITO:</div>
          <div>
            <span style={{ color: '#f43f5e' }}>let </span>
            <span style={{ color: '#fb923c', fontWeight: 'bold' }}>mut </span>
            <span style={{ color: '#f8fafc' }}>epoca</span>
            <span style={{ color: '#94a3b8' }}>: </span>
            <span style={{ color: '#38bdf8' }}>usize </span>
            <span style={{ color: '#94a3b8' }}>= </span>
            <span style={{ color: '#34d399' }}>0</span>
            <span style={{ color: '#94a3b8' }}>;</span>
          </div>
          <div style={{ marginTop: 12 }}>
            <span style={{ color: '#f8fafc' }}>epoca </span>
            <span style={{ color: '#34d399' }}>+= </span>
            <span style={{ color: '#34d399' }}>1</span>
            <span style={{ color: '#94a3b8' }}>;</span>
          </div>

          <div
            style={{
              opacity: checkOpacity,
              marginTop: 24,
              padding: 16,
              backgroundColor: 'rgba(16, 185, 129, 0.15)',
              borderLeft: '4px solid #10b981',
              borderRadius: 8,
              color: '#6ee7b7',
              fontSize: 22,
            }}
          >
            ✅ <strong>Compilado con éxito:</strong> El compilador sabe exactamente dónde cambia la memoria.
          </div>
        </CodeCard>
      </div>

      <div
        style={{
          opacity: checkOpacity,
          marginTop: 35,
          textAlign: 'center',
          color: '#cbd5e1',
          fontSize: 28,
          maxWidth: 820,
        }}
      >
        ⚡ Ideal para contadores de época, optimizadores como SGD y cálculo de loss.
      </div>
    </AbsoluteFill>
  );
};

// Escena 4: Shadowing
const SceneShadowing = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const cardScale = spring({ frame, fps, config: { damping: 14 } });
  const badgeOpacity = interpolate(frame, [25, 40], [0, 1], { extrapolateRight: 'clamp' });

  return (
    <AbsoluteFill
      style={{
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        alignItems: 'center',
        padding: 60,
      }}
    >
      <div
        style={{
          color: '#c084fc',
          fontSize: 26,
          fontWeight: 'bold',
          marginBottom: 20,
          letterSpacing: 1.5,
        }}
      >
        3. SHADOWING (SOMBREADO) EN PIPELINES
      </div>

      <div style={{ transform: `scale(${cardScale})`, width: '100%', maxWidth: 920 }}>
        <CodeCard title="data_loader.rs" borderColor="#a855f7">
          <div style={{ color: '#94a3b8' }}>// Transforma datos reutilizando el nombre:</div>
          <div>
            <span style={{ color: '#f43f5e' }}>let </span>
            <span style={{ color: '#f8fafc' }}>batch_size </span>
            <span style={{ color: '#94a3b8' }}>= </span>
            <span style={{ color: '#fde047' }}>"64"</span>
            <span style={{ color: '#94a3b8' }}>; </span>
            <span style={{ color: '#64748b' }}>// &str (del archivo config)</span>
          </div>
          <div style={{ marginTop: 10 }}>
            <span style={{ color: '#f43f5e' }}>let </span>
            <span style={{ color: '#f8fafc' }}>batch_size</span>
            <span style={{ color: '#94a3b8' }}>: </span>
            <span style={{ color: '#38bdf8' }}>usize </span>
            <span style={{ color: '#94a3b8' }}>= </span>
            <span style={{ color: '#34d399' }}>64</span>
            <span style={{ color: '#94a3b8' }}>; </span>
            <span style={{ color: '#64748b' }}>// usize (número real)</span>
          </div>
          <div style={{ marginTop: 10 }}>
            <span style={{ color: '#f43f5e' }}>let </span>
            <span style={{ color: '#f8fafc' }}>batch_size </span>
            <span style={{ color: '#94a3b8' }}>= </span>
            <span style={{ color: '#f8fafc' }}>batch_size </span>
            <span style={{ color: '#94a3b8' }}>* </span>
            <span style={{ color: '#34d399' }}>2</span>
            <span style={{ color: '#94a3b8' }}>; </span>
            <span style={{ color: '#64748b' }}>// 128 (escalado)</span>
          </div>
        </CodeCard>
      </div>

      <div
        style={{
          opacity: badgeOpacity,
          marginTop: 35,
          textAlign: 'center',
          color: '#cbd5e1',
          fontSize: 28,
          maxWidth: 820,
        }}
      >
        💎 <strong>Adiós a variables feas:</strong> Ya no necesitas `batch_str`, `batch_int` ni `batch_final`.
      </div>
    </AbsoluteFill>
  );
};

// Escena 5: Resumen y Outro
const SceneOutro = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const scale = spring({ frame, fps, config: { damping: 12 } });

  return (
    <AbsoluteFill
      style={{
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        alignItems: 'center',
        padding: 60,
        textAlign: 'center',
      }}
    >
      <div
        style={{
          transform: `scale(${scale})`,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: 24,
        }}
      >
        <div style={{ fontSize: 90 }}>🦀 🧠</div>
        <h2 style={{ fontSize: 62, fontWeight: 900, color: 'white', margin: 0 }}>
          ¡Clase 01 Dominada!
        </h2>
        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            gap: 16,
            textAlign: 'left',
            backgroundColor: '#1e293b',
            padding: '30px 45px',
            borderRadius: 20,
            border: '1px solid #334155',
            marginTop: 10,
          }}
        >
          <div style={{ fontSize: 26, color: '#f8fafc' }}>🛡️ Inmutabilidad = Seguridad Multi-GPU</div>
          <div style={{ fontSize: 26, color: '#f8fafc' }}>⚡ `let mut` = Control explícito de estados</div>
          <div style={{ fontSize: 26, color: '#f8fafc' }}>💎 Shadowing = Pipelines de datos limpios</div>
        </div>
        <div style={{ fontSize: 24, color: '#94a3b8', marginTop: 15 }}>
          Siguiente: <strong>Clase 02 — Tipos de Datos Escalares y Compuestos</strong>
        </div>
      </div>
    </AbsoluteFill>
  );
};

// Componente Principal
export const Clase01Video = () => {
  const frame = useCurrentFrame();
  const { durationInFrames } = useVideoConfig();

  // Barra de progreso inferior
  const progress = (frame / durationInFrames) * 100;

  return (
    <AbsoluteFill
      style={{
        backgroundColor: '#090d16',
        fontFamily: 'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
        color: '#ffffff',
        position: 'relative',
        overflow: 'hidden',
      }}
    >
      {/* Fondo con brillo sutil */}
      <div
        style={{
          position: 'absolute',
          top: -200,
          left: -200,
          width: 700,
          height: 700,
          borderRadius: '50%',
          background: 'radial-gradient(circle, rgba(249, 115, 22, 0.15) 0%, transparent 70%)',
          filter: 'blur(80px)',
        }}
      />
      <div
        style={{
          position: 'absolute',
          bottom: -200,
          right: -200,
          width: 700,
          height: 700,
          borderRadius: '50%',
          background: 'radial-gradient(circle, rgba(56, 189, 248, 0.12) 0%, transparent 70%)',
          filter: 'blur(80px)',
        }}
      />

      {/* Marca de agua superior */}
      <div
        style={{
          position: 'absolute',
          top: 50,
          left: 60,
          right: 60,
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          zIndex: 10,
        }}
      >
        <span style={{ fontSize: 22, fontWeight: 'bold', color: '#64748b' }}>
          CURSO: RUST PARA IA
        </span>
        <span style={{ fontSize: 20, color: '#475569', fontFamily: 'monospace' }}>
          FASE 1 • 01_VARIABLES
        </span>
      </div>

      {/* Secuencia de escenas (360 frames = 12 segundos @ 30 fps) */}
      <Sequence from={0} durationInFrames={75}>
        <SceneIntro />
      </Sequence>

      <Sequence from={75} durationInFrames={75}>
        <SceneInmutabilidad />
      </Sequence>

      <Sequence from={150} durationInFrames={75}>
        <SceneMutabilidad />
      </Sequence>

      <Sequence from={225} durationInFrames={75}>
        <SceneShadowing />
      </Sequence>

      <Sequence from={300} durationInFrames={60}>
        <SceneOutro />
      </Sequence>

      {/* Barra de progreso inferior */}
      <div
        style={{
          position: 'absolute',
          bottom: 0,
          left: 0,
          height: 8,
          width: `${progress}%`,
          background: 'linear-gradient(to right, #f97316, #38bdf8)',
          zIndex: 50,
        }}
      />
    </AbsoluteFill>
  );
};
