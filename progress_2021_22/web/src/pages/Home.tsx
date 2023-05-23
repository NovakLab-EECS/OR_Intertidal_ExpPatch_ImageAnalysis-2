import '../styles/Home.css'
import Button from '../components/Button'
import { useNavigate } from 'react-router-dom'
import { atom } from 'jotai'
import { useEffect } from 'react'

const App = () => {

  const navigate = useNavigate()

  const folders = atom([])
  const photos = atom([])



  return (
    <div className="App">
      <header className="App-header">
        <div className="title">
          <h1>Verify Inferences and Bounding Boxes</h1>
          <h3>Capstone Group CS72 - Serial Image Analysis</h3>
        </div>
        <div className="buttons">
          <Button onClick={() => {
            navigate('/inferences')
          }}>
            Verify Inferences
          </Button>
          <Button onClick={() => {
            navigate('/boundingBoxes')
          }}>
            Draw Bounding Boxes
          </Button>
        </div>
      </header>
    </div>
  )
}

export default App
