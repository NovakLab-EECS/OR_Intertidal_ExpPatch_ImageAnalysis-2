import { faArrowLeft } from '@fortawesome/free-solid-svg-icons';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { FC, useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import Button from '../components/Button';

import '../styles/BBoxes.css'

import DATA_JSON from '../public/data.json';

const BBoxes: FC = () => {
  const navigate = useNavigate()

  const getFolders = async () => {
    return Object.keys(DATA_JSON);
  }

  const [folders, setFolders] = useState<string[]>([])

  useEffect(() => {
    getFolders().then(setFolders)
  }, [])


  return (
    <>
      <header>
        <div className="back">
          <Button onClick={() => {
            navigate('/')
          }}>
            <FontAwesomeIcon icon={faArrowLeft} style={{
              paddingRight: '4px'
            }} />
            Back
          </Button>
        </div>
        <div className="title">
          <h1>Bounding Boxes</h1>
        </div>
        <div className="content">
          {folders.map(folder => (
            <div key={folder} className="folder">
              <h3 onClick={(e) => {
                e.stopPropagation()
                navigate(`/bboxes/${folder}`)
              }}>
                {folder}
              </h3>
            </div>
          ))}
        </div>
      </header>
    </>
  )
}

export default BBoxes