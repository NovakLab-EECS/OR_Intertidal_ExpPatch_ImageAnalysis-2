import { faArrowLeft } from '@fortawesome/free-solid-svg-icons';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { FC, useEffect, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import Button from '../components/Button';
import LazyLoader from '../components/Image';

import '../styles/Draw.css'

import DATA_JSON from '../public/data.json';
import { LazyLoadImage } from 'react-lazy-load-image-component';


interface IPhoto {
  name: string;
  url: string;
}

const GITHUB_BASE = 'https://github.com/NovakLab-EECS/OR_Intertidal_ExpPatch_ImageAnalysis/raw/master/ExpPatch-Pics/ExpPatchPics-Processed/'

const Draw: FC = () => {
  const navigate = useNavigate()
  const params = useParams()

  const [photos, setPhotos] = useState<IPhoto[]>([])
  const [error, setError] = useState<string>()

  const getFolder = async (fName: string) => {
    // @ts-expect-error
    const files = DATA_JSON[fName] as string[]
    setPhotos(() => {
      return files.map(file => {
        return {
          name: file,
          url: `${GITHUB_BASE}${fName}/${file}`
        }
      })
    })
  }

  useEffect(() => {
    getFolder(params.folder as string)
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
          <h1>Draw Boxes</h1>
        </div>
        <div className="content">
          <div className="description">
            <p>The goal is to incase each individual species in its own bounding box</p>
          </div>
          {error && <div className="error">{error}</div>}
          <div className="photos">
            {photos.map(photo => (
              <div key={photo.name} className="photo">
                <LazyLoadImage src={photo.url} alt={photo.name} width="250px" effect="blur" onClick={() => {
                  navigate(`/draw/${params.folder}/${photo.name.split('.')[0]}`)
                }} />
                <small>{photo.name}</small>
              </div>
            ))}
          </div>

        </div>
      </header>
    </>
  )
}

export default Draw