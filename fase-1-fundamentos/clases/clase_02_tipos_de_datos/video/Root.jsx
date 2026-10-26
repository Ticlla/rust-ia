import React from 'react';
import { Composition } from 'remotion';
import { Clase02Video } from './Clase02Video';

export const Root = () => {
  return (
    <Composition
      id="Clase02"
      component={Clase02Video}
      durationInFrames={360}
      fps={30}
      width={1080}
      height={1920}
    />
  );
};
