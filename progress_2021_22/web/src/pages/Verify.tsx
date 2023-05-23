import { faArrowLeft } from '@fortawesome/free-solid-svg-icons';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { FC, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import BBViz from '../components/BBViz';
import Button from '../components/Button';
import Dropzone from '../components/Dropzone';

import DATA_JSON from '../public/data.json';
const GITHUB_BASE =
  'https://github.com/NovakLab-EECS/OR_Intertidal_ExpPatch_ImageAnalysis/raw/master/ExpPatch-Pics/ExpPatchPics-Processed/';

import '../styles/verify.css';

const Verify: FC = () => {
  const navigate = useNavigate();

  const [loaded, setLoaded] = useState<boolean>(false);

  const [objs, setObjs] = useState<any[]>([]);
  const [lastHovered, setLastHovered] = useState<Record<string, number>>({});

  const IMAGE_WIDTH = 1000;
  // Calculate the height of the image based on the width (4/3 ratio)
  const IMAGE_HEIGHT = (IMAGE_WIDTH * 3) / 4;

  const onDrop = (files: File[]) => {
    const [file] = files;

    const fr = new FileReader();

    fr.onload = () => {
      const data = fr.result as string;
      const result = JSON.parse(data);

      const res = Object.keys(result).map((key) => {
        const value = result[key];
        const image = getImage(key);

        return {
          image,
          value,
          key,
        };
      });

      const lastHoveredSetup = res.reduce((acc, curr) => {
        const { key } = curr;
        if (lastHovered) {
          // @ts-ignore
          acc[key] = -1;
        }
        return acc;
      }, {});

      setLastHovered(lastHoveredSetup);
      setObjs(res);
      setLoaded(true);
    };

    fr.readAsText(file);
  };

  const getImage = (fName: string): string | undefined => {
    const folders = Object.keys(DATA_JSON);
    const betterFolders = folders.map((folder) => folder.split('_')[0]);

    const betterFName = fName.split('_')[0];

    for (let i = 0; i < betterFolders.length; i++) {
      if (betterFolders[i] === betterFName) {
        const folder = folders[i] as keyof typeof DATA_JSON;

        const str = DATA_JSON[folder].filter(
          (image) => image.split('.')[0] === fName
        )[0];

        return `${GITHUB_BASE}${folder}/${str}`;
      }
    }
    return undefined;
  };

  return (
    <>
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
          <h1>Verify Inferences</h1>
        </div>
        <div className="main">
          <div className="description">
            <p>
              Upload the generated JSON file below to begin verification of the
              bounding boxes that were generated.
            </p>
          </div>
          <div
            className="dropzone"
            style={{
              display: loaded ? 'none' : 'flex',
            }}
          >
            <div className="upload">
              <Dropzone
                callback={(files) => {
                  onDrop(files);
                }}
                // Only accept JSON files
                accept={{
                  'application/json': ['.json'],
                }}
              />
            </div>
          </div>
          <div>
            {objs && (
              <div
                className="results"
                style={{
                  margin: '0 30px',
                  display: 'flex',
                  flexDirection: 'column',
                }}
              >
                {objs.map((obj) => {
                  const { image, value, key } = obj;
                  return (
                    <div
                      key={key}
                      style={{
                        display: 'flex',
                        flexDirection: 'column',
                        justifyContent: 'center',
                      }}
                    >
                      <BBViz
                        image={image}
                        imageSize={[IMAGE_WIDTH, IMAGE_HEIGHT]}
                        boxes={value.map(
                          (box: {
                            x: number;
                            y: number;
                            width: number;
                            height: number;
                            label: string | number;
                          }) => {
                            let x = box.x * IMAGE_WIDTH;
                            let y = box.y * IMAGE_HEIGHT;
                            let w = box.width * IMAGE_WIDTH;
                            let h = box.height * IMAGE_HEIGHT;

                            // The x and the y are going to be the middle of the box,
                            // we need to convert them to the top left corner
                            x -= w / 2;
                            y -= h / 2;

                            return [x, y, w, h, box.label].map((v, i) => {
                              if (i !== 4 && typeof v === 'number')
                                return Math.round(v);
                              return v;
                            });
                          }
                        )}
                        onSelected={(boxIndex) => {
                          setLastHovered({
                            ...lastHovered,
                            [key]: boxIndex,
                          });
                        }}
                        onClick={(e) => {
                          e.preventDefault();

                          const wow = objs[lastHovered[key]];
                          console.log(lastHovered[key], key, lastHovered);
                        }}
                      />
                      <span
                        style={{
                          textAlign: 'center',
                        }}
                      >
                        {key}
                      </span>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>
      </header>
    </>
  );
};

export default Verify;
