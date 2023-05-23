import { faArrowLeft } from '@fortawesome/free-solid-svg-icons';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { FC, useEffect, useRef, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import Button from '../components/Button';
import BBox from '../components/BBox';

import DATA_JSON from '../public/data.json';
const GITHUB_BASE =
  'https://github.com/NovakLab-EECS/OR_Intertidal_ExpPatch_ImageAnalysis/raw/master/ExpPatch-Pics/ExpPatchPics-Processed/';

const DrawBB = () => {
  const navigate = useNavigate();
  const params = useParams();

  const getURL = (fName: string, iName: string) => {
    const folders = Object.keys(DATA_JSON);
    const betterFolders = folders.map((folder) => folder.split('_')[0]);

    const betterFName = fName.split('_')[0];

    for (let i = 0; i < betterFolders.length; i++) {
      if (betterFolders[i] === betterFName) {
        const folder = folders[i] as keyof typeof DATA_JSON;

        const str = DATA_JSON[folder].filter(
          (image) => image.split('.')[0] === iName
        );

        return `${GITHUB_BASE}${folder}/${str}`;
      }
    }
    return undefined;
  };

  return (
    <header>
      <div className="back">
        <Button
          onClick={() => {
            navigate('/');
          }}
        >
          <FontAwesomeIcon
            icon={faArrowLeft}
            style={{
              paddingRight: '4px',
            }}
          />
          Back
        </Button>
      </div>
      <div className="title">
        <h1>Bounding Boxes for {params.image}</h1>
      </div>
      <div className="content">
        <BBox
          imgSrc={getURL(params.folder || '', params.image || '') || ''}
          imgSize={{
            height: 768,
            width: 1024,
          }}
          types={[
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
            20, 21, 22, 23, 24, 25,
          ]}
          fName={params.image || ''}
        />
      </div>
    </header>
  );
};

export default DrawBB;
