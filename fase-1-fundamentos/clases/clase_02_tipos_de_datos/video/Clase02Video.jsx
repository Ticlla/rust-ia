import React from 'react';
import {
  AbsoluteFill,
  interpolate,
  Sequence,
  spring,
  useCurrentFrame,
  useVideoConfig,
} from 'remotion';

// Tarjeta de código estilo macOS
const CodeCard = ({ title, children, borderColor = '#334155' }) => (
  <div
    style={{
      backgroundColor: '#0f172a',
      borderRadius: 24,
      border: `2px solid ${borderColor}`,
      boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.7)',
      overflow: 'hidden',
      width: '100%',
      maxWidth: 940,
    }}
  >
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
    <div style={{ padding: 32, fontFamily: 'monospace', fontSize: 28, lineHeight: 1.6 }}>
      {children}
    </div>
  </div>
);

// Escena 1: Intro / Hook
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
          background: 'rgba(56, 189, 248, 0.2)',
          border: '2px solid #38bdf8',
          color: '#38bdf8',
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
        Tipos de Datos y{' '}
        <span
          style={{
            background: 'linear-gradient(to right, #38bdf8, #a855f7)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
          }}
        >
          Casting Estricto
        </span>
      </h1>

      <p
        style={{
          opacity: descOpacity,
          fontSize: 32,
          color: '#94a3b8',
          maxWidth: 820,
        }}
      >
        Clase 02: Por qué Rust no te deja sumar enteros con flotantes y cómo protege tus cálculos en IA
      </p>
    </AbsoluteFill>
  );
};

// Escena 2: Escalares esenciales en IA
const SceneEscalares = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const cardScale = spring({ frame, fps, config: { damping: 14 } });
  const badgeOpacity = interpolate(frame, [20, 35], [0, 1], { extrapolateRight: 'clamp' });

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
        1. ESCALARES CLAVE EN INTELIGENCIA ARTIFICIAL
      </div>

      <div style={{ transform: `scale(${cardScale})`, width: '100%', maxWidth: 940 }}>
        <CodeCard title="types_ai.rs" borderColor="#38bdf8">
          <div>
            <span style={{ color: '#f43f5e' }}>let </span>
            <span style={{ color: '#f8fafc' }}>pixel</span>
            <span style={{ color: '#94a3b8' }}>: </span>
            <span style={{ color: '#38bdf8' }}>u8 </span>
            <span style={{ color: '#94a3b8' }}>= </span>
            <span style={{ color: '#34d399' }}>255</span>
            <span style={{ color: '#94a3b8' }}>; </span>
            <span style={{ color: '#64748b' }}>// Píxeles RGB (0-255)</span>
          </div>
          <div style={{ marginTop: 10 }}>
            <span style={{ color: '#f43f5e' }}>let </span>
            <span style={{ color: '#f8fafc' }}>peso</span>
            <span style={{ color: '#94a3b8' }}>: </span>
            <span style={{ color: '#38bdf8' }}>f32 </span>
            <span style={{ color: '#94a3b8' }}>= </span>
            <span style={{ color: '#34d399' }}>0.8542</span>
            <span style={{ color: '#94a3b8' }}>; </span>
            <span style={{ color: '#64748b' }}>// El rey de GPUs / tensores</span>
          </div>
          <div style={{ marginTop: 10 }}>
            <span style={{ color: '#f43f5e' }}>let </span>
            <span style={{ color: '#f8fafc' }}>idx</span>
            <span style={{ color: '#94a3b8' }}>: </span>
            <span style={{ color: '#38bdf8' }}>usize </span>
            <span style={{ color: '#94a3b8' }}>= </span>
            <span style={{ color: '#34d399' }}>0</span>
            <span style={{ color: '#94a3b8' }}>; </span>
            <span style={{ color: '#64748b' }}>// Índices y tamaños de memoria</span>
          </div>
          <div style={{ marginTop: 10 }}>
            <span style={{ color: '#f43f5e' }}>let </span>
            <span style={{ color: '#f8fafc' }}>mask</span>
            <span style={{ color: '#94a3b8' }}>: </span>
            <span style={{ color: '#38bdf8' }}>bool </span>
            <span style={{ color: '#94a3b8' }}>= </span>
            <span style={{ color: '#fb923c' }}>true</span>
            <span style={{ color: '#94a3b8' }}>; </span>
            <span style={{ color: '#64748b' }}>// Máscara de Transformers</span>
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
        💡 En Rust cada byte cuenta: <strong>`f32`</strong> reduce a la mitad el consumo de VRAM frente al <strong>`f64`</strong> de Python.
      </div>
    </AbsoluteFill>
  );
};

// Escena 3: Casting Estricto con 'as'
const SceneCasting = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const cardScale = spring({ frame, fps, config: { damping: 14 } });
  const alertOpacity = interpolate(frame, [25, 40], [0, 1], { extrapolateRight: 'clamp' });

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
          color: '#f43f5e',
          fontSize: 26,
          fontWeight: 'bold',
          marginBottom: 20,
          letterSpacing: 1.5,
        }}
      >
        2. CASTING ESTRICTO: CERO CONVERSIONES MÁGICAS
      </div>

      <div style={{ transform: `scale(${cardScale})`, width: '100%', maxWidth: 940 }}>
        <CodeCard title="metric_loss.rs" borderColor="#f43f5e">
          <div>
            <span style={{ color: '#f43f5e' }}>let </span>
            <span style={{ color: '#f8fafc' }}>total_loss</span>
            <span style={{ color: '#94a3b8' }}>: </span>
            <span style={{ color: '#38bdf8' }}>f32 </span>
            <span style={{ color: '#94a3b8' }}>= </span>
            <span style={{ color: '#34d399' }}>45.8</span>
            <span style={{ color: '#94a3b8' }}>;</span>
          </div>
          <div style={{ marginTop: 8 }}>
            <span style={{ color: '#f43f5e' }}>let </span>
            <span style={{ color: '#f8fafc' }}>batches</span>
            <span style={{ color: '#94a3b8' }}>: </span>
            <span style={{ color: '#38bdf8' }}>usize </span>
            <span style={{ color: '#94a3b8' }}>= </span>
            <span style={{ color: '#34d399' }}>10</span>
            <span style={{ color: '#94a3b8' }}>;</span>
          </div>

          <div style={{ marginTop: 18, color: '#ef4444' }}>
            // ❌ total_loss / batches; (Error de compilación E0308)
          </div>

          <div style={{ marginTop: 10 }}>
            <span style={{ color: '#f43f5e' }}>let </span>
            <span style={{ color: '#f8fafc' }}>promedio </span>
            <span style={{ color: '#94a3b8' }}>= </span>
            <span style={{ color: '#f8fafc' }}>total_loss </span>
            <span style={{ color: '#94a3b8' }}>/ (</span>
            <span style={{ color: '#f8fafc' }}>batches </span>
            <span style={{ color: '#fb923c', fontWeight: 'bold' }}>as </span>
            <span style={{ color: '#38bdf8' }}>f32</span>
            <span style={{ color: '#94a3b8' }}>); </span>
            <span style={{ color: '#34d399' }}>// ✅ 4.58</span>
          </div>

          <div
            style={{
              opacity: alertOpacity,
              marginTop: 20,
              padding: 16,
              backgroundColor: 'rgba(244, 63, 94, 0.15)',
              borderLeft: '4px solid #f43f5e',
              borderRadius: 8,
              color: '#fecdd3',
              fontSize: 22,
            }}
          >
            🛡️ Sin redondeos silenciosos ni desbordamientos imprevistos en tu loss.
          </div>
        </CodeCard>
      </div>

      <div
        style={{
          opacity: alertOpacity,
          marginTop: 35,
          textAlign: 'center',
          color: '#cbd5e1',
          fontSize: 28,
          maxWidth: 820,
        }}
      >
        En Python los tipos se convierten automáticamente. En Rust <strong>tú tienes el control absoluto de la precisión numérica</strong>.
      </div>
    </AbsoluteFill>
  );
};

// Escena 4: Tuplas y Arrays fijos
const SceneTuplasArrays = () => {
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
          color: '#a855f7',
          fontSize: 26,
          fontWeight: 'bold',
          marginBottom: 20,
          letterSpacing: 1.5,
        }}
      >
        3. TUPLAS Y ARRAYS FIJOS EN EL STACK
      </div>

      <div style={{ transform: `scale(${cardScale})`, width: '100%', maxWidth: 940 }}>
        <CodeCard title="model_eval.rs" borderColor="#a855f7">
          <div style={{ color: '#94a3b8' }}>// Tupla: agrupa tipos distintos (métricas del modelo)</div>
          <div>
            <span style={{ color: '#f43f5e' }}>let </span>
            <span style={{ color: '#f8fafc' }}>metricas</span>
            <span style={{ color: '#94a3b8' }}>: (&str, </span>
            <span style={{ color: '#38bdf8' }}>f32</span>
            <span style={{ color: '#94a3b8' }}>, </span>
            <span style={{ color: '#38bdf8' }}>usize</span>
            <span style={{ color: '#94a3b8' }}>) = (</span>
            <span style={{ color: '#fde047' }}>"ResNet-50"</span>
            <span style={{ color: '#94a3b8' }}>, </span>
            <span style={{ color: '#34d399' }}>0.942</span>
            <span style={{ color: '#94a3b8' }}>, </span>
            <span style={{ color: '#34d399' }}>50</span>
            <span style={{ color: '#94a3b8' }}>);</span>
          </div>

          <div style={{ marginTop: 18, color: '#94a3b8' }}>
            // Array [T; N]: memoria contigua en el STACK (cero asignación en heap)
          </div>
          <div>
            <span style={{ color: '#f43f5e' }}>let </span>
            <span style={{ color: '#f8fafc' }}>shape_tensor</span>
            <span style={{ color: '#94a3b8' }}>: [</span>
            <span style={{ color: '#38bdf8' }}>usize</span>
            <span style={{ color: '#94a3b8' }}>; </span>
            <span style={{ color: '#34d399' }}>3</span>
            <span style={{ color: '#94a3b8' }}>] = [</span>
            <span style={{ color: '#34d399' }}>32</span>
            <span style={{ color: '#94a3b8' }}>, </span>
            <span style={{ color: '#34d399' }}>224</span>
            <span style={{ color: '#94a3b8' }}>, </span>
            <span style={{ color: '#34d399' }}>224</span>
            <span style={{ color: '#94a3b8' }}>];</span>
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
        🚀 <strong>Alto rendimiento:</strong> Los arrays con tamaño fijo `[T; N]` residen en memoria ultrarrápida (Stack), perfectos para formas de tensores.
      </div>
    </AbsoluteFill>
  );
};

// Escena 5: Outro / Resumen
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
        <div style={{ fontSize: 90 }}>🦀 ⚡</div>
        <h2 style={{ fontSize: 62, fontWeight: 900, color: 'white', margin: 0 }}>
          ¡Clase 02 Dominada!
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
          <div style={{ fontSize: 26, color: '#f8fafc' }}>🎯 Escalares: `f32` para Deep Learning y `u8` para imágenes</div>
          <div style={{ fontSize: 26, color: '#f8fafc' }}>🛡️ Casting con `as`: cero errores de precisión silenciosos</div>
          <div style={{ fontSize: 26, color: '#f8fafc' }}>📦 Tuplas y Arrays: memoria contigua sin costo de Heap</div>
        </div>
        <div style={{ fontSize: 24, color: '#94a3b8', marginTop: 15 }}>
          Siguiente: <strong>Clase 03 — Funciones y Paso de Parámetros en IA</strong>
        </div>
      </div>
    </AbsoluteFill>
  );
};

// Componente Principal
export const Clase02Video = () => {
  const frame = useCurrentFrame();
  const { durationInFrames } = useVideoConfig();

  // Progreso en porcentaje para la barra inferior
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
      {/* Luces de fondo difusas */}
      <div
        style={{
          position: 'absolute',
          top: -200,
          right: -200,
          width: 700,
          height: 700,
          borderRadius: '50%',
          background: 'radial-gradient(circle, rgba(56, 189, 248, 0.15) 0%, transparent 70%)',
          filter: 'blur(80px)',
        }}
      />
      <div
        style={{
          position: 'absolute',
          bottom: -200,
          left: -200,
          width: 700,
          height: 700,
          borderRadius: '50%',
          background: 'radial-gradient(circle, rgba(168, 85, 247, 0.15) 0%, transparent 70%)',
          filter: 'blur(80px)',
        }}
      />

      {/* Header superior */}
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
          FASE 1 • 02_TIPOS_DE_DATOS
        </span>
      </div>

      {/* Secuencias de escenas */}
      <Sequence from={0} durationInFrames={75}>
        <SceneIntro />
      </Sequence>

      <Sequence from={75} durationInFrames={75}>
        <SceneEscalares />
      </Sequence>

      <Sequence from={150} durationInFrames={75}>
        <SceneCasting />
      </Sequence>

      <Sequence from={225} durationInFrames={75}>
        <SceneTuplasArrays />
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
          background: 'linear-gradient(to right, #38bdf8, #a855f7)',
          zIndex: 50,
        }}
      />
    </AbsoluteFill>
  );
};
