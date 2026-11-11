/**
 * Header.jsx
 * ----------
 * Top navigation bar with backend health indicator.
 */

export default function Header({ backendOnline }) {
  return (
    <header className="bg-white border-b border-gray-200 shadow-sm">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo + title */}
          <div className="flex items-center gap-3">
            <div className="flex items-center justify-center w-9 h-9 rounded-lg bg-teal-600 text-white font-bold text-sm select-none">
              T
            </div>
            <div>
              <h1 className="text-lg font-semibold text-gray-900 leading-tight">
                Tanita Human Digital Twin
              </h1>
              <p className="text-xs text-gray-500 leading-tight">
                3D Body Mesh Reconstruction &mdash; POC
              </p>
            </div>
          </div>

          {/* Backend status badge */}
          <div className="flex items-center gap-2">
            <span
              className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium ${
                backendOnline
                  ? 'bg-green-50 text-green-700 ring-1 ring-green-200'
                  : 'bg-red-50 text-red-700 ring-1 ring-red-200'
              }`}
            >
              <span
                className={`w-1.5 h-1.5 rounded-full ${
                  backendOnline ? 'bg-green-500' : 'bg-red-500'
                }`}
              />
              {backendOnline ? 'Backend online' : 'Backend offline'}
            </span>
          </div>
        </div>
      </div>
    </header>
  )
}
