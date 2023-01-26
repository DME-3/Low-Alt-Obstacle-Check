from shapely.geometry.polygon import Polygon

rhein_coords = [[6.975930879168959,50.95657408794192],
[6.972535124899244,50.95458641663587],
[6.967809452697489,50.95085449055784],
[6.964303258235203,50.94648787705209],
[6.962946874051212,50.94281719236779],
[6.96352513940893,50.93620396292537],
[6.965306071776009,50.93028081346332],
[6.969701292329919,50.91962775943225],
[6.974670336848934,50.91147192727765],
[6.979500669602354,50.90559075596518],
[6.988910832741286,50.89811010115705],
[7.001299970098163,50.89290819036373],
[7.011608685904386,50.89253183470269],
[7.013331310567981,50.89615626376607],
[7.001160198607601,50.89750937089659],
[6.991533142700823,50.90103254802452],
[6.988667339298418,50.90270433843045],
[6.983479655838492,50.90851708398598],
[6.975279465952471,50.91906362354751],
[6.970153101800916,50.93012572612207],
[6.970658769054046,50.93156686745481],
[6.968078026948048,50.93770005507804],
[6.968636570741024,50.94353672207991],
[6.972110561853963,50.94977637503464],
[6.978430041779998,50.95436079405016],
[6.984223870632302,50.9521389418798],
[6.994914076799508,50.95996315483612],
[6.997507875246713,50.96355172063916],
[6.993574311759938,50.96491142024293],
[6.975930879168959,50.95657408794192]]

cologne_coords = [[6.959171759112499,50.99430853610677],
[6.901929300555015,50.99758667765417],
[6.860725537056751,50.97815312645849],
[6.871070534707544,50.95391768475689],
[6.853180990552776,50.95154510132253],
[6.851595175933216,50.92580526907562],
[6.876727291355323,50.92464428516085],
[6.900600612823995,50.90568428194248],
[6.938992081005395,50.88954543485961],
[6.970909450690675,50.88870549136099],
[6.971514736431644,50.8509754372153],
[7.011695608648989,50.86350468618333],
[7.022704592201563,50.87133048443056],
[7.038277772878323,50.87428005523759],
[7.016151872258929,50.88948819170199],
[7.046725536152215,50.89781497439955],
[7.017412050633681,50.91656406216891],
[7.033468790229806,50.93691718823904],
[7.023598510031402,50.95380479296748],
[7.007444206265411,50.97007849952217],
[6.959171759112499,50.99430853610677]]

rhein_polygon = Polygon(rhein_coords)
cologne_polygon = Polygon(cologne_coords)