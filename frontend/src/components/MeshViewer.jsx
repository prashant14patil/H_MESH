/**
 * MeshViewer.jsx
 * --------------
 * Interactive 3D viewer for the reconstructed SMPL OBJ mesh.
 * Uses React Three Fiber + @react-three/drei.
 *
 * Props:
 *   meshUrl: string   – absolute URL of the OBJ file served by the FastAPI backend
 *   onLoad(info):     – called with { vertices, faces } once the mesh is parsed
 *   onError(msg):     – called with an error string if loading fails
 */

import { Suspense, useRef, useCallback, Component } from 'react'
import { Canvas, useLoader } from '@react-three/fiber'
import { OrbitControls, Center } from '@react-three/drei'
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader'
import * as THREE from 'three'

// ---------------------------------------------------------------------------
// Inner mesh component (inside Canvas, has access to Three.js context)
// ---------------------------------------------------------------------------
function OBJMesh({ url, onLoad, onError }) {
  const obj = useLoader(OBJLoader, url)

  // Compute vertex/face counts and call onLoad once
  const reported = useRef(false)
  if (!reported.current && obj) {
    reported.current = true
    let vertices = 0
    let faces = 0
    obj.traverse((child) => {
      if (child.isMesh && child.geometry) {
        const pos = child.geometry.attributes.position
        if (pos) vertices += pos.count
        const idx = child.geometry.index
        if (idx) faces += Math.floor(idx.count / 3)
        else if (pos) faces += Math.floor(pos.count / 3)
      }
    })
    onLoad?.({ vertices, faces })
  }

  // Apply a clean material to all meshes in the OBJ
  obj.traverse((child) => {
    if (child.isMesh) {
      child.material = new THREE.MeshStandardMaterial({
        color: '#4db6ac',
        roughness: 0.6,
        metalness: 0.05,
        side: THREE.FrontSide,
      })
      child.castShadow = true
      child.receiveShadow = true
    }
  })

  return (
    <Center>
      {/* Rotate 180° around X to correct up-axis mismatch (OBJ exported with
          Z-up / head-down orientation) so the mesh stands upright facing
          forward in Three.js's Y-up coordinate system. */}
      <group rotation={[Math.PI, 0, 0]}>
        <primitive object={obj} />
      </group>
    </Center>
  )
}

// ---------------------------------------------------------------------------
// Error boundary for loader errors
// ---------------------------------------------------------------------------
class MeshErrorBoundary extends Component {
  constructor(props) {
    super(props)
    this.state = { hasError: false, message: '' }
  }
  static getDerivedStateFromError(error) {
    return { hasError: true, message: error?.message || 'Unknown error' }
  }
  componentDidCatch(error) {
    this.props.onError?.(error?.message || 'Failed to load mesh')
  }
  render() {
    if (this.state.hasError) return null
    return this.props.children
  }
}

// ---------------------------------------------------------------------------
// Loading placeholder inside canvas
// ---------------------------------------------------------------------------
function LoadingMesh() {
  return (
    <mesh>
      <sphereGeometry args={[0.5, 16, 16]} />
      <meshBasicMaterial color="#e0f2f1" wireframe />
    </mesh>
  )
}

// ---------------------------------------------------------------------------
// Public component
// ---------------------------------------------------------------------------
export default function MeshViewer({ meshUrl, onLoad, onError }) {
  const controlsRef = useRef(null)

  const handleReset = useCallback(() => {
    controlsRef.current?.reset()
  }, [])

  if (!meshUrl) return null

  return (
    <div className="relative w-full rounded-xl overflow-hidden bg-gray-900 shadow-inner" style={{ height: 520 }}>
      {/* Three.js canvas */}
      <Canvas
        camera={{ position: [0, 0.8, 2.5], fov: 45, near: 0.01, far: 100 }}
        shadows
        gl={{ antialias: true }}
        style={{ background: 'linear-gradient(180deg, #0f1117 0%, #1a2030 100%)' }}
      >
        {/* Lighting */}
        <ambientLight intensity={0.4} />
        <directionalLight
          position={[3, 8, 5]}
          intensity={1.2}
          castShadow
          shadow-mapSize={[1024, 1024]}
        />
        <directionalLight position={[-3, 4, -2]} intensity={0.4} />
        <pointLight position={[0, -2, 2]} intensity={0.3} color="#80cbc4" />

        {/* Mesh */}
        <MeshErrorBoundary onError={onError}>
          <Suspense fallback={<LoadingMesh />}>
            <OBJMesh url={meshUrl} onLoad={onLoad} onError={onError} />
          </Suspense>
        </MeshErrorBoundary>

        {/* Controls */}
        <OrbitControls
          ref={controlsRef}
          enableDamping
          dampingFactor={0.08}
          minDistance={0.5}
          maxDistance={8}
          target={[0, 0, 0]}
        />
      </Canvas>

      {/* Overlay controls */}
      <div className="absolute bottom-3 right-3 flex flex-col gap-2">
        <button
          onClick={handleReset}
          title="Reset camera"
          className="flex items-center justify-center w-8 h-8 rounded-lg bg-white/10 hover:bg-white/20 text-white backdrop-blur-sm transition-colors"
        >
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round"
              d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0l3.181 3.183a8.25 8.25 0 0013.803-3.7M4.031 9.865a8.25 8.25 0 0113.803-3.7l3.181 3.182m0-4.991v4.99" />
          </svg>
        </button>
      </div>

      {/* Usage hint */}
      <div className="absolute bottom-3 left-3">
        <p className="text-xs text-white/40">
          Drag to rotate · Scroll to zoom · Right-drag to pan
        </p>
      </div>

      {/* Loading overlay — shown while Suspense is active */}
      <div
        className="absolute inset-0 flex flex-col items-center justify-center bg-gray-900/80 pointer-events-none"
        id="mesh-loading-overlay"
        style={{ display: 'none' }}
      >
        <svg className="w-8 h-8 animate-spin text-teal-400 mb-3" fill="none" viewBox="0 0 24 24">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
        </svg>
        <p className="text-sm text-white/60">Loading mesh…</p>
      </div>
    </div>
  )
}
